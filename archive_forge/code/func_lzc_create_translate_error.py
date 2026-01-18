import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_create_translate_error(ret, name, ds_type, props):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_name(name)
        raise lzc_exc.PropertyInvalid(name)
    if ret == errno.EEXIST:
        raise lzc_exc.FilesystemExists(name)
    if ret == errno.ENOENT:
        raise lzc_exc.ParentNotFound(name)
    raise _generic_exception(ret, name, 'Failed to create filesystem')