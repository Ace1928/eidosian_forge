import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_destroy_snaps_translate_errors(ret, errlist, snaps, defer):
    if ret == 0:
        return

    def _map(ret, name):
        if ret == errno.EEXIST:
            return lzc_exc.SnapshotIsCloned(name)
        if ret == errno.ENOENT:
            return lzc_exc.PoolNotFound(name)
        if ret == errno.EBUSY:
            return lzc_exc.SnapshotIsHeld(name)
        return _generic_exception(ret, name, 'Failed to destroy snapshot')
    _handle_err_list(ret, errlist, snaps, lzc_exc.SnapshotDestructionFailure, _map)