import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_snapshot_translate_errors(ret, errlist, snaps, props):
    if ret == 0:
        return

    def _map(ret, name):
        if ret == errno.EXDEV:
            pool_names = map(_pool_name, snaps)
            same_pool = all((x == pool_names[0] for x in pool_names))
            if same_pool:
                return lzc_exc.DuplicateSnapshots(name)
            else:
                return lzc_exc.PoolsDiffer(name)
        elif ret == errno.EINVAL:
            if any((not _is_valid_snap_name(s) for s in snaps)):
                return lzc_exc.NameInvalid(name)
            elif any((len(s) > MAXNAMELEN for s in snaps)):
                return lzc_exc.NameTooLong(name)
            else:
                return lzc_exc.PropertyInvalid(name)
        if ret == errno.EEXIST:
            return lzc_exc.SnapshotExists(name)
        if ret == errno.ENOENT:
            return lzc_exc.FilesystemNotFound(name)
        return _generic_exception(ret, name, 'Failed to create snapshot')
    _handle_err_list(ret, errlist, snaps, lzc_exc.SnapshotFailure, _map)