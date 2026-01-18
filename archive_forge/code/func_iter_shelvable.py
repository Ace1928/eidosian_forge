import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def iter_shelvable(self):
    """Iterable of tuples describing shelvable changes.

        As well as generating the tuples, this updates several members.
        Tuples may be::

           ('add file', file_id, work_kind, work_path)
           ('delete file', file_id, target_kind, target_path)
           ('rename', file_id, target_path, work_path)
           ('change kind', file_id, target_kind, work_kind, target_path)
           ('modify text', file_id)
           ('modify target', file_id, target_target, work_target)
        """
    for change in self.iter_changes:
        if change.kind[0] is None and change.name[1] == '':
            continue
        if change.kind[1] is None and change.name[0] == '':
            continue
        if change.kind[0] is None or change.versioned[0] is False:
            self.creation[change.file_id] = (change.kind[1], change.name[1], change.parent_id[1], change.versioned)
            yield ('add file', change.file_id, change.kind[1], change.path[1])
        elif change.kind[1] is None or change.versioned[0] is False:
            self.deletion[change.file_id] = (change.kind[0], change.name[0], change.parent_id[0], change.versioned)
            yield ('delete file', change.file_id, change.kind[0], change.path[0])
        else:
            if change.name[0] != change.name[1] or change.parent_id[0] != change.parent_id[1]:
                self.renames[change.file_id] = (change.name, change.parent_id)
                yield (('rename', change.file_id) + change.path)
            if change.kind[0] != change.kind[1]:
                yield ('change kind', change.file_id, change.kind[0], change.kind[1], change.path[0])
            elif change.kind[0] == 'symlink':
                t_target = self.target_tree.get_symlink_target(change.path[0])
                w_target = self.work_tree.get_symlink_target(change.path[1])
                yield ('modify target', change.file_id, change.path[0], t_target, w_target)
            elif change.changed_content:
                yield ('modify text', change.file_id)