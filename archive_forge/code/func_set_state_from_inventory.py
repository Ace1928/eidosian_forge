import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def set_state_from_inventory(self, new_inv):
    """Set new_inv as the current state.

        This API is called by tree transform, and will usually occur with
        existing parent trees.

        :param new_inv: The inventory object to set current state from.
        """
    if 'evil' in debug.debug_flags:
        trace.mutter_callsite(1, 'set_state_from_inventory called; please mutate the tree instead')
    tracing = 'dirstate' in debug.debug_flags
    if tracing:
        trace.mutter('set_state_from_inventory trace:')
    self._read_dirblocks_if_needed()
    new_iterator = new_inv.iter_entries_by_dir()
    old_iterator = iter(list(self._iter_entries()))
    current_new = next(new_iterator)
    current_old = next(old_iterator)

    def advance(iterator):
        try:
            return next(iterator)
        except StopIteration:
            return None
    while current_new or current_old:
        if current_old and current_old[1][0][0] in (b'a', b'r'):
            current_old = advance(old_iterator)
            continue
        if current_new:
            new_path_utf8 = current_new[0].encode('utf8')
            new_dirname, new_basename = osutils.split(new_path_utf8)
            new_id = current_new[1].file_id
            new_entry_key = (new_dirname, new_basename, new_id)
            current_new_minikind = DirState._kind_to_minikind[current_new[1].kind]
            if current_new_minikind == b't':
                fingerprint = current_new[1].reference_revision or b''
            else:
                fingerprint = b''
        else:
            new_path_utf8 = new_dirname = new_basename = new_id = new_entry_key = None
        if not current_old:
            if tracing:
                trace.mutter("Appending from new '%s'.", new_path_utf8.decode('utf8'))
            self.update_minimal(new_entry_key, current_new_minikind, executable=current_new[1].executable, path_utf8=new_path_utf8, fingerprint=fingerprint, fullscan=True)
            current_new = advance(new_iterator)
        elif not current_new:
            if tracing:
                trace.mutter("Truncating from old '%s/%s'.", current_old[0][0].decode('utf8'), current_old[0][1].decode('utf8'))
            self._make_absent(current_old)
            current_old = advance(old_iterator)
        elif new_entry_key == current_old[0]:
            if current_old[1][0][3] != current_new[1].executable or current_old[1][0][0] != current_new_minikind:
                if tracing:
                    trace.mutter("Updating in-place change '%s'.", new_path_utf8.decode('utf8'))
                self.update_minimal(current_old[0], current_new_minikind, executable=current_new[1].executable, path_utf8=new_path_utf8, fingerprint=fingerprint, fullscan=True)
            current_old = advance(old_iterator)
            current_new = advance(new_iterator)
        elif lt_by_dirs(new_dirname, current_old[0][0]) or (new_dirname == current_old[0][0] and new_entry_key[1:] < current_old[0][1:]):
            if tracing:
                trace.mutter("Inserting from new '%s'.", new_path_utf8.decode('utf8'))
            self.update_minimal(new_entry_key, current_new_minikind, executable=current_new[1].executable, path_utf8=new_path_utf8, fingerprint=fingerprint, fullscan=True)
            current_new = advance(new_iterator)
        else:
            if tracing:
                trace.mutter("Deleting from old '%s/%s'.", current_old[0][0].decode('utf8'), current_old[0][1].decode('utf8'))
            self._make_absent(current_old)
            current_old = advance(old_iterator)
    self._mark_modified()
    self._id_index = None
    self._packed_stat_index = None
    if tracing:
        trace.mutter('set_state_from_inventory complete.')