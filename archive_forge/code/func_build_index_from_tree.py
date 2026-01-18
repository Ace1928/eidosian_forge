import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def build_index_from_tree(root_path: Union[str, bytes], index_path: Union[str, bytes], object_store: ObjectContainer, tree_id: bytes, honor_filemode: bool=True, validate_path_element=validate_path_element_default, symlink_fn=None):
    """Generate and materialize index from a tree.

    Args:
      tree_id: Tree to materialize
      root_path: Target dir for materialized index files
      index_path: Target path for generated index
      object_store: Non-empty object store holding tree contents
      honor_filemode: An optional flag to honor core.filemode setting in
        config file, default is core.filemode=True, change executable bit
      validate_path_element: Function to validate path elements to check
        out; default just refuses .git and .. directories.

    Note: existing index is wiped and contents are not merged
        in a working dir. Suitable only for fresh clones.
    """
    index = Index(index_path, read=False)
    if not isinstance(root_path, bytes):
        root_path = os.fsencode(root_path)
    for entry in iter_tree_contents(object_store, tree_id):
        if not validate_path(entry.path, validate_path_element):
            continue
        full_path = _tree_to_fs_path(root_path, entry.path)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        if S_ISGITLINK(entry.mode):
            if not os.path.isdir(full_path):
                os.mkdir(full_path)
            st = os.lstat(full_path)
        else:
            obj = object_store[entry.sha]
            assert isinstance(obj, Blob)
            st = build_file_from_blob(obj, entry.mode, full_path, honor_filemode=honor_filemode, symlink_fn=symlink_fn)
        if not honor_filemode or S_ISGITLINK(entry.mode):
            st_tuple = (entry.mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid, st.st_size, st.st_atime, st.st_mtime, st.st_ctime)
            st = st.__class__(st_tuple)
        index[entry.path] = index_entry_from_stat(st, entry.sha)
    index.write()