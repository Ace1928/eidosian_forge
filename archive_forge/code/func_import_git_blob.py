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
def import_git_blob(texts, mapping, path, name, hexshas, base_bzr_tree, parent_id, revision_id, parent_bzr_trees, lookup_object, modes, store_updater, lookup_file_id):
    """Import a git blob object into a bzr repository.

    :param texts: VersionedFiles to add to
    :param path: Path in the tree
    :param blob: A git blob
    :return: Inventory delta for this file
    """
    if not isinstance(path, bytes):
        raise TypeError(path)
    decoded_path = decode_git_path(path)
    base_mode, mode = modes
    base_hexsha, hexsha = hexshas
    if mapping.is_special_file(path):
        return []
    if base_hexsha == hexsha and base_mode == mode:
        return []
    file_id = lookup_file_id(decoded_path)
    if stat.S_ISLNK(mode):
        cls = InventoryLink
    else:
        cls = InventoryFile
    ie = cls(file_id, decode_git_path(name), parent_id)
    if ie.kind == 'file':
        ie.executable = mode_is_executable(mode)
    if base_hexsha == hexsha and mode_kind(base_mode) == mode_kind(mode):
        base_exec = base_bzr_tree.is_executable(decoded_path)
        if ie.kind == 'symlink':
            ie.symlink_target = base_bzr_tree.get_symlink_target(decoded_path)
        else:
            ie.text_size = base_bzr_tree.get_file_size(decoded_path)
            ie.text_sha1 = base_bzr_tree.get_file_sha1(decoded_path)
        if ie.kind == 'symlink' or ie.executable == base_exec:
            ie.revision = base_bzr_tree.get_file_revision(decoded_path)
        else:
            blob = lookup_object(hexsha)
    else:
        blob = lookup_object(hexsha)
        if ie.kind == 'symlink':
            ie.revision = None
            ie.symlink_target = decode_git_path(blob.data)
        else:
            ie.text_size = sum(map(len, blob.chunked))
            ie.text_sha1 = osutils.sha_strings(blob.chunked)
    parent_keys = []
    for ptree in parent_bzr_trees:
        intertree = InterTree.get(ptree, base_bzr_tree)
        try:
            ppath = intertree.find_source_paths(decoded_path, recurse='none')
        except NoSuchFile:
            continue
        if ppath is None:
            continue
        pkind = ptree.kind(ppath)
        if pkind == ie.kind and (pkind == 'symlink' and ptree.get_symlink_target(ppath) == ie.symlink_target or (pkind == 'file' and ptree.get_file_sha1(ppath) == ie.text_sha1 and (ptree.is_executable(ppath) == ie.executable))):
            ie.revision = ptree.get_file_revision(ppath)
            break
        parent_key = (file_id, ptree.get_file_revision(ppath))
        if parent_key not in parent_keys:
            parent_keys.append(parent_key)
    if ie.revision is None:
        ie.revision = revision_id
        if ie.revision is None:
            raise ValueError('no file revision set')
        if ie.kind == 'symlink':
            chunks = []
        else:
            chunks = blob.chunked
        texts.insert_record_stream([ChunkedContentFactory((file_id, ie.revision), tuple(parent_keys), ie.text_sha1, chunks)])
    invdelta = []
    if base_hexsha is not None:
        old_path = decoded_path
        if stat.S_ISDIR(base_mode):
            invdelta.extend(remove_disappeared_children(base_bzr_tree, old_path, lookup_object(base_hexsha), [], lookup_object))
    else:
        old_path = None
    invdelta.append((old_path, decoded_path, file_id, ie))
    if base_hexsha != hexsha:
        store_updater.add_object(blob, (ie.file_id, ie.revision), path)
    return invdelta