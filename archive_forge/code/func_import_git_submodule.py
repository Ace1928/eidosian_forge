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
def import_git_submodule(texts, mapping, path, name, hexshas, base_bzr_tree, parent_id, revision_id, parent_bzr_trees, lookup_object, modes, store_updater, lookup_file_id):
    """Import a git submodule."""
    base_hexsha, hexsha = hexshas
    base_mode, mode = modes
    if base_hexsha == hexsha and base_mode == mode:
        return ([], {})
    path = decode_git_path(path)
    file_id = lookup_file_id(path)
    invdelta = []
    ie = TreeReference(file_id, decode_git_path(name), parent_id)
    ie.revision = revision_id
    if base_hexsha is not None:
        old_path = path
        if stat.S_ISDIR(base_mode):
            invdelta.extend(remove_disappeared_children(base_bzr_tree, old_path, lookup_object(base_hexsha), [], lookup_object))
    else:
        old_path = None
    ie.reference_revision = mapping.revision_id_foreign_to_bzr(hexsha)
    texts.insert_record_stream([ChunkedContentFactory((file_id, ie.revision), (), None, [])])
    invdelta.append((old_path, path, file_id, ie))
    return (invdelta, {})