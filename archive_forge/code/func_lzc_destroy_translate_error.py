import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_destroy_translate_error(ret, name):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(name)
    if ret == errno.ENOENT:
        raise lzc_exc.FilesystemNotFound(name)
    raise _generic_exception(ret, name, 'Failed to destroy dataset')