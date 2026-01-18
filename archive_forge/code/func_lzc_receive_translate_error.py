import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def lzc_receive_translate_error(ret, snapname, fd, force, origin, props):
    if ret == 0:
        return
    if ret == errno.EINVAL:
        if not _is_valid_snap_name(snapname) and (not _is_valid_fs_name(snapname)):
            raise lzc_exc.NameInvalid(snapname)
        elif len(snapname) > MAXNAMELEN:
            raise lzc_exc.NameTooLong(snapname)
        elif origin is not None and (not _is_valid_snap_name(origin)):
            raise lzc_exc.NameInvalid(origin)
        else:
            raise lzc_exc.BadStream()
    if ret == errno.ENOENT:
        if not _is_valid_snap_name(snapname):
            raise lzc_exc.NameInvalid(snapname)
        else:
            raise lzc_exc.DatasetNotFound(snapname)
    if ret == errno.EEXIST:
        raise lzc_exc.DatasetExists(snapname)
    if ret == errno.ENOTSUP:
        raise lzc_exc.StreamFeatureNotSupported()
    if ret == errno.ENODEV:
        raise lzc_exc.StreamMismatch(_fs_name(snapname))
    if ret == errno.ETXTBSY:
        raise lzc_exc.DestinationModified(_fs_name(snapname))
    if ret == errno.EBUSY:
        raise lzc_exc.DatasetBusy(_fs_name(snapname))
    if ret == errno.ENOSPC:
        raise lzc_exc.NoSpace(_fs_name(snapname))
    if ret == errno.EDQUOT:
        raise lzc_exc.QuotaExceeded(_fs_name(snapname))
    if ret == errno.ENAMETOOLONG:
        raise lzc_exc.NameTooLong(snapname)
    if ret == errno.EROFS:
        raise lzc_exc.ReadOnlyPool(_pool_name(snapname))
    if ret == errno.EAGAIN:
        raise lzc_exc.SuspendedPool(_pool_name(snapname))
    raise lzc_exc.StreamIOError(ret)