import os
from ctypes import (
from ctypes.wintypes import DWORD, LONG
def get_processtree_pids(pid, include_parent=True):
    """Return a list with all the pids of a process tree"""
    parents = get_all_processes_pids()
    all_pids = list(parents.keys())
    pids = {pid}
    while 1:
        pids_new = pids.copy()
        for _pid in all_pids:
            if parents[_pid] in pids:
                pids_new.add(_pid)
        if pids_new == pids:
            break
        pids = pids_new.copy()
    if not include_parent:
        pids.remove(pid)
    return list(pids)