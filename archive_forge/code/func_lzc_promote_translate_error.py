import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_promote_translate_error(ret, name):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(name)
        raise lzc_exc.NotClone(name)
    if ret == errno.ENOTSOCK:
        raise lzc_exc.NotClone(name)
    if ret == errno.ENOENT:
        raise lzc_exc.FilesystemNotFound(name)
    if ret == errno.EEXIST:
        raise lzc_exc.SnapshotExists(name)
    raise _generic_exception(ret, name, 'Failed to promote dataset')