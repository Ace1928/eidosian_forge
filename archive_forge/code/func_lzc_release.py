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
def lzc_release(holds):
    """
    Release *user holds* on snapshots.

    If the snapshot has been marked for
    deferred destroy (by lzc_destroy_snaps(defer=B_TRUE)), it does not have
    any clones, and all the user holds are removed, then the snapshot will be
    destroyed.

    The snapshots must all be in the same pool.

    :param holds: a ``dict`` where keys are snapshot names and values are
                  lists of hold tags to remove.
    :type holds: dict of bytes : list of bytes
    :return: a list of any snapshots that do not exist and of any tags that do not
             exist for existing snapshots.
             Such tags are qualified with a corresponding snapshot name
             using the following format :file:`{pool}/{fs}@{snap}#{tag}`
    :rtype: list of bytes

    :raises HoldReleaseFailure: if one or more existing holds could not be released.

    Holds which failed to release because they didn't exist will have an entry
    added to errlist, but will not cause an overall failure.

    This call is success if ``holds`` was empty or all holds that
    existed, were successfully removed.
    Otherwise an exception will be raised.
    """
    errlist = {}
    holds_dict = {}
    for snap, hold_list in holds.iteritems():
        if not isinstance(hold_list, list):
            raise TypeError('holds must be in a list')
        holds_dict[snap] = {hold: None for hold in hold_list}
    nvlist = nvlist_in(holds_dict)
    with nvlist_out(errlist) as errlist_nvlist:
        ret = _lib.lzc_release(nvlist, errlist_nvlist)
    errors.lzc_release_translate_errors(ret, errlist, holds)
    assert all((x == errno.ENOENT for x in errlist.itervalues()))
    return errlist.keys()