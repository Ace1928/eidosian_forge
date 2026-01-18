import errno
import os
import posixpath
import stat
from collections import deque
from functools import partial
from io import BytesIO
from typing import Union, List, Tuple, Set
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.errors import NotTreeError
from dulwich.index import (Index, IndexEntry, blob_from_path_and_stat,
from dulwich.object_store import OverlayObjectStore, iter_tree_contents, BaseObjectStore
from dulwich.objects import S_IFGITLINK, S_ISGITLINK, ZERO_SHA, Blob, Tree, ObjectID
from .. import controldir as _mod_controldir
from .. import delta, errors, mutabletree, osutils, revisiontree, trace
from .. import transport as _mod_transport
from .. import tree as _mod_tree
from .. import urlutils, workingtree
from ..bzr.inventorytree import InventoryTreeChange
from ..revision import CURRENT_REVISION, NULL_REVISION
from ..transport import get_transport
from ..tree import MissingNestedTree, TreeEntry
from .mapping import (decode_git_path, default_mapping, encode_git_path,
def snapshot_workingtree(target: MutableGitIndexTree, want_unversioned: bool=False) -> Tuple[ObjectID, Set[bytes]]:
    """Snapshot a working tree into a tree object."""
    extras = set()
    blobs = {}
    dirified = []
    trust_executable = target._supports_executable()
    for path, index_entry in target._recurse_index_entries():
        index_entry = getattr(index_entry, 'this', index_entry)
        try:
            live_entry = target._live_entry(path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                blobs[path] = (ZERO_SHA, 0)
            else:
                raise
        else:
            if live_entry is None:
                if S_ISGITLINK(index_entry.mode):
                    blobs[path] = (index_entry.sha, index_entry.mode)
                else:
                    dirified.append((path, Tree().id, stat.S_IFDIR))
                    target.store.add_object(Tree())
            else:
                mode = live_entry.mode
                if not trust_executable:
                    if mode_is_executable(index_entry.mode):
                        mode |= 73
                    else:
                        mode &= ~73
                if live_entry.sha != index_entry.sha:
                    rp = decode_git_path(path)
                    if stat.S_ISREG(live_entry.mode):
                        blob = Blob()
                        with target.get_file(rp) as f:
                            blob.data = f.read()
                    elif stat.S_ISLNK(live_entry.mode):
                        blob = Blob()
                        blob.data = os.fsencode(target.get_symlink_target(rp))
                    else:
                        blob = None
                    if blob is not None:
                        target.store.add_object(blob)
                blobs[path] = (live_entry.sha, cleanup_mode(live_entry.mode))
    if want_unversioned:
        for extra in target._iter_files_recursive(include_dirs=False):
            extra, accessible = osutils.normalized_filename(extra)
            np = encode_git_path(extra)
            if np in blobs:
                continue
            st = target._lstat(extra)
            obj: Union[Tree, Blob]
            if stat.S_ISDIR(st.st_mode):
                obj = Tree()
            elif stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
                obj = blob_from_path_and_stat(os.fsencode(target.abspath(extra)), st)
            else:
                continue
            target.store.add_object(obj)
            blobs[np] = (obj.id, cleanup_mode(st.st_mode))
            extras.add(np)
    return (commit_tree(target.store, dirified + [(p, s, m) for p, (s, m) in blobs.items()]), extras)