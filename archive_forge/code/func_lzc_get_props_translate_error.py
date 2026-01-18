import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_get_props_translate_error(ret, name):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        _validate_fs_or_snap_name(name)
    if ret == errno.ENOENT:
        raise lzc_exc.DatasetNotFound(name)
    raise _generic_exception(ret, name, 'Failed to get properties')