import errno
import os
import re
import stat
import tarfile
import zipfile
from io import BytesIO
from . import urlutils
from .bzr import generate_ids
from .controldir import ControlDir, is_control_filename
from .errors import BzrError, CommandError, NotBranchError
from .osutils import (basename, file_iterator, file_kind, isdir, pathjoin,
from .trace import warning
from .transform import resolve_conflicts
from .transport import NoSuchFile, get_transport
from .workingtree import WorkingTree
def import_archive_to_transform(tree, archive_file, tt):
    prefix = common_directory(names_of_files(archive_file))
    removed = set()
    for path, entry in tree.iter_entries_by_dir():
        if entry.parent_id is None:
            continue
        trans_id = tt.trans_id_tree_path(path)
        tt.delete_contents(trans_id)
        removed.add(path)
    added = set()
    implied_parents = set()
    seen = set()
    for member in archive_file.getmembers():
        if member.type == 'g':
            continue
        relative_path = member.name
        if not isinstance(relative_path, str):
            relative_path = relative_path.decode('utf-8')
        if prefix is not None:
            relative_path = relative_path[len(prefix) + 1:]
            relative_path = relative_path.rstrip('/')
        if relative_path == '':
            continue
        if should_ignore(relative_path):
            continue
        add_implied_parents(implied_parents, relative_path)
        trans_id = tt.trans_id_tree_path(relative_path)
        added.add(relative_path.rstrip('/'))
        path = tree.abspath(relative_path)
        if member.name in seen:
            if tt.final_kind(trans_id) == 'file':
                tt.set_executability(None, trans_id)
            tt.cancel_creation(trans_id)
        seen.add(member.name)
        if member.isreg():
            tt.create_file(file_iterator(archive_file.extractfile(member)), trans_id)
            executable = member.mode & 73 != 0
            tt.set_executability(executable, trans_id)
        elif member.isdir():
            do_directory(tt, trans_id, tree, relative_path, path)
        elif member.issym():
            tt.create_symlink(member.linkname, trans_id)
        else:
            continue
        if not tt.final_is_versioned(trans_id):
            name = basename(member.name.rstrip('/'))
            file_id = generate_ids.gen_file_id(name)
            tt.version_file(trans_id, file_id=file_id)
    for relative_path in implied_parents.difference(added):
        if relative_path == '':
            continue
        trans_id = tt.trans_id_tree_path(relative_path)
        path = tree.abspath(relative_path)
        do_directory(tt, trans_id, tree, relative_path, path)
        if tt.tree_file_id(trans_id) is None:
            tt.version_file(trans_id, file_id=trans_id)
        added.add(relative_path)
    for path in removed.difference(added):
        tt.unversion_file(tt.trans_id_tree_path(path))
    for conflict in tt.cook_conflicts(resolve_conflicts(tt)):
        warning(conflict)