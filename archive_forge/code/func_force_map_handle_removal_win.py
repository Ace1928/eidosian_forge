from .util import (
import sys
from functools import reduce
def force_map_handle_removal_win(self, base_path):
    """ONLY AVAILABLE ON WINDOWS
        On windows removing files is not allowed if anybody still has it opened.
        If this process is ourselves, and if the whole process uses this memory
        manager (as far as the parent framework is concerned) we can enforce
        closing all memory maps whose path matches the given base path to
        allow the respective operation after all.
        The respective system must NOT access the closed memory regions anymore !
        This really may only be used if you know that the items which keep
        the cursors alive will not be using it anymore. They need to be recreated !
        :return: Amount of closed handles

        **Note:** does nothing on non-windows platforms"""
    if sys.platform != 'win32':
        return
    num_closed = 0
    for path, rlist in self._fdict.items():
        if path.startswith(base_path):
            for region in rlist:
                region.release()
                num_closed += 1
    return num_closed