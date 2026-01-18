import errno
import functools
import fcntl
import os
import struct
import threading
from . import exceptions
from . import _error_translation as errors
from .bindings import libzfs_core
from ._constants import MAXNAMELEN
from .ctypes import int32_t
from ._nvlist import nvlist_in, nvlist_out
def lzc_hold(holds, fd=None):
    """
    Create *user holds* on snapshots.  If there is a hold on a snapshot,
    the snapshot can not be destroyed.  (However, it can be marked for deletion
    by :func:`lzc_destroy_snaps` ( ``defer`` = `True` ).)

    :param holds: the dictionary of names of the snapshots to hold mapped to the hold names.
    :type holds: dict of bytes : bytes
    :type fd: int or None
    :param fd: if not None then it must be the result of :func:`os.open` called as ``os.open("/dev/zfs", O_EXCL)``.
    :type fd: int or None
    :return: a list of the snapshots that do not exist.
    :rtype: list of bytes

    :raises HoldFailure: if a hold was impossible on one or more of the snapshots.
    :raises BadHoldCleanupFD: if ``fd`` is not a valid file descriptor associated with :file:`/dev/zfs`.

    The snapshots must all be in the same pool.

    If ``fd`` is not None, then when the ``fd`` is closed (including on process
    termination), the holds will be released.  If the system is shut down
    uncleanly, the holds will be released when the pool is next opened
    or imported.

    Holds for snapshots which don't exist will be skipped and have an entry
    added to the return value, but will not cause an overall failure.
    No exceptions is raised if all holds, for snapshots that existed, were succesfully created.
    Otherwise :exc:`.HoldFailure` exception is raised and no holds will be created.
    :attr:`.HoldFailure.errors` may contain a single element for an error that is not
    specific to any hold / snapshot, or it may contain one or more elements
    detailing specific error per each affected hold.
    """
    errlist = {}
    if fd is None:
        fd = -1
    nvlist = nvlist_in(holds)
    with nvlist_out(errlist) as errlist_nvlist:
        ret = _lib.lzc_hold(nvlist, fd, errlist_nvlist)
    errors.lzc_hold_translate_errors(ret, errlist, holds, fd)
    assert all((x == errno.ENOENT for x in errlist.itervalues()))
    return errlist.keys()