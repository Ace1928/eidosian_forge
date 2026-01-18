import ctypes
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
def set_entries_in_acl(self, entry_count, p_explicit_entry_list, p_old_acl):
    """Merge new ACEs into an existing ACL, returning a new ACL."""
    pp_new_acl = self._get_void_pp()
    self._win32_utils.run_and_check_output(advapi32.SetEntriesInAclW, entry_count, p_explicit_entry_list, p_old_acl, pp_new_acl)
    return pp_new_acl