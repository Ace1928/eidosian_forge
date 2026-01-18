import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_rollback_translate_error(ret, name):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(name)
        raise lzc_exc.SnapshotNotFound(name)
    if ret == errno.ENOENT:
        if not _is_valid_fs_name(name):
            raise lzc_exc.NameInvalid(name)
        else:
            raise lzc_exc.FilesystemNotFound(name)
    raise _generic_exception(ret, name, 'Failed to rollback')