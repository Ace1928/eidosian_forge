import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_rename_translate_error(ret, source, target):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(source)
        _validate_fs_name(target)
        if _pool_name(source) != _pool_name(target):
            raise lzc_exc.PoolsDiffer(source)
    if ret == errno.EEXIST:
        raise lzc_exc.FilesystemExists(target)
    if ret == errno.ENOENT:
        raise lzc_exc.FilesystemNotFound(source)
    raise _generic_exception(ret, source, 'Failed to rename dataset')