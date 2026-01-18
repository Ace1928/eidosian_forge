import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_clone_translate_error(ret, name, origin, props):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(name)
        _validate_snap_name(origin)
        if _pool_name(name) != _pool_name(origin):
            raise lzc_exc.PoolsDiffer(name)
        else:
            raise lzc_exc.PropertyInvalid(name)
    if ret == errno.EEXIST:
        raise lzc_exc.FilesystemExists(name)
    if ret == errno.ENOENT:
        if not _is_valid_snap_name(origin):
            raise lzc_exc.SnapshotNameInvalid(origin)
        raise lzc_exc.DatasetNotFound(name)
    raise _generic_exception(ret, name, 'Failed to create clone')