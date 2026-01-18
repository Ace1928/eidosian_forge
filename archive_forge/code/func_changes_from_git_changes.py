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
def changes_from_git_changes(changes, mapping, specific_files=None, include_unchanged=False, source_extras=None, target_extras=None):
    """Create a iter_changes-like generator from a git stream.

    source and target are iterators over tuples with:
        (filename, sha, mode)
    """
    if target_extras is None:
        target_extras = set()
    if source_extras is None:
        source_extras = set()
    for change_type, old, new in changes:
        if change_type == 'unchanged' and (not include_unchanged):
            continue
        oldpath, oldmode, oldsha = old
        newpath, newmode, newsha = new
        if oldpath is not None:
            oldpath_decoded = decode_git_path(oldpath)
        else:
            oldpath_decoded = None
        if newpath is not None:
            newpath_decoded = decode_git_path(newpath)
        else:
            newpath_decoded = None
        if not (specific_files is None or (oldpath_decoded is not None and osutils.is_inside_or_parent_of_any(specific_files, oldpath_decoded)) or (newpath_decoded is not None and osutils.is_inside_or_parent_of_any(specific_files, newpath_decoded))):
            continue
        if oldpath is not None and mapping.is_special_file(oldpath):
            continue
        if newpath is not None and mapping.is_special_file(newpath):
            continue
        if oldpath is None:
            oldexe = None
            oldkind = None
            oldname = None
            oldparent = None
            oldversioned = False
        else:
            oldversioned = oldpath not in source_extras
            if oldmode:
                oldexe = mode_is_executable(oldmode)
                oldkind = mode_kind(oldmode)
            else:
                oldexe = False
                oldkind = None
            if oldpath_decoded == '':
                oldparent = None
                oldname = ''
            else:
                oldparentpath, oldname = osutils.split(oldpath_decoded)
                oldparent = mapping.generate_file_id(oldparentpath)
        if newpath is None:
            newexe = None
            newkind = None
            newname = None
            newparent = None
            newversioned = False
        else:
            newversioned = newpath not in target_extras
            if newmode:
                newexe = mode_is_executable(newmode)
                newkind = mode_kind(newmode)
            else:
                newexe = False
                newkind = None
            if newpath_decoded == '':
                newparent = None
                newname = ''
            else:
                newparentpath, newname = osutils.split(newpath_decoded)
                newparent = mapping.generate_file_id(newparentpath)
        if not include_unchanged and oldkind == 'directory' and (newkind == 'directory') and (oldpath_decoded == newpath_decoded):
            continue
        if oldversioned and change_type != 'copy':
            fileid = mapping.generate_file_id(oldpath_decoded)
        elif newversioned:
            fileid = mapping.generate_file_id(newpath_decoded)
        else:
            fileid = None
        if oldkind == 'directory' and newkind == 'directory':
            modified = False
        else:
            modified = oldsha != newsha or oldmode != newmode
        yield InventoryTreeChange(fileid, (oldpath_decoded, newpath_decoded), modified, (oldversioned, newversioned), (oldparent, newparent), (oldname, newname), (oldkind, newkind), (oldexe, newexe), copied=change_type == 'copy')