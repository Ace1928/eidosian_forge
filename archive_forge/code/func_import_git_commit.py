import posixpath
import stat
from dulwich.object_store import tree_lookup_path
from dulwich.objects import (S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Commit, Tag,
from .. import debug, errors, osutils, trace
from ..bzr.inventory import (InventoryDirectory, InventoryFile, InventoryLink,
from ..bzr.inventorytree import InventoryRevisionTree
from ..bzr.testament import StrictTestament3
from ..bzr.versionedfile import ChunkedContentFactory
from ..errors import BzrError
from ..revision import NULL_REVISION
from ..transport import NoSuchFile
from ..tree import InterTree
from ..tsort import topo_sort
from .mapping import (DEFAULT_FILE_MODE, decode_git_path, mode_is_executable,
from .object_store import LRUTreeCache, _tree_to_objects
def import_git_commit(repo, mapping, head, lookup_object, target_git_object_retriever, trees_cache, strict):
    o = lookup_object(head)
    rev, roundtrip_revid, verifiers = mapping.import_commit(o, mapping.revision_id_foreign_to_bzr, strict)
    if roundtrip_revid is not None:
        original_revid = rev.revision_id
        rev.revision_id = roundtrip_revid
    parent_trees = trees_cache.revision_trees(rev.parent_ids)
    ensure_inventories_in_repo(repo, parent_trees)
    if parent_trees == []:
        base_bzr_tree = trees_cache.revision_tree(NULL_REVISION)
        base_tree = None
        base_mode = None
    else:
        base_bzr_tree = parent_trees[0]
        base_tree = lookup_object(o.parents[0]).tree
        base_mode = stat.S_IFDIR
    store_updater = target_git_object_retriever._get_updater(rev)
    inv_delta, unusual_modes = import_git_tree(repo.texts, mapping, b'', b'', (base_tree, o.tree), base_bzr_tree, None, rev.revision_id, parent_trees, lookup_object, (base_mode, stat.S_IFDIR), store_updater, mapping.generate_file_id, allow_submodules=repo._format.supports_tree_reference)
    if unusual_modes != {}:
        for path, mode in unusual_modes.iteritems():
            warn_unusual_mode(rev.foreign_revid, path, mode)
        mapping.import_unusual_file_modes(rev, unusual_modes)
    try:
        basis_id = rev.parent_ids[0]
    except IndexError:
        basis_id = NULL_REVISION
        base_bzr_inventory = None
    else:
        base_bzr_inventory = base_bzr_tree.root_inventory
    rev.inventory_sha1, inv = repo.add_inventory_by_delta(basis_id, inv_delta, rev.revision_id, rev.parent_ids, base_bzr_inventory)
    ret_tree = InventoryRevisionTree(repo, inv, rev.revision_id)
    if verifiers and roundtrip_revid is not None:
        testament = StrictTestament3(rev, ret_tree)
        calculated_verifiers = {'testament3-sha1': testament.as_sha1()}
        if calculated_verifiers != verifiers:
            trace.mutter('Testament SHA1 %r for %r did not match %r.', calculated_verifiers['testament3-sha1'], rev.revision_id, verifiers['testament3-sha1'])
            rev.revision_id = original_revid
            rev.inventory_sha1, inv = repo.add_inventory_by_delta(basis_id, inv_delta, rev.revision_id, rev.parent_ids, base_bzr_tree)
            ret_tree = InventoryRevisionTree(repo, inv, rev.revision_id)
    else:
        calculated_verifiers = {}
    store_updater.add_object(o, calculated_verifiers, None)
    store_updater.finish()
    trees_cache.add(ret_tree)
    repo.add_revision(rev.revision_id, rev)
    if 'verify' in debug.debug_flags:
        verify_commit_reconstruction(target_git_object_retriever, lookup_object, o, rev, ret_tree, parent_trees, mapping, unusual_modes, verifiers)