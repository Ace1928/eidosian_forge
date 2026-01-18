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
def lzc_receive(snapname, fd, force=False, origin=None, props=None):
    """
    Receive from the specified ``fd``, creating the specified snapshot.

    :param bytes snapname: the name of the snapshot to create.
    :param int fd: the file descriptor from which to read the stream.
    :param bool force: whether to roll back or destroy the target filesystem
                       if that is required to receive the stream.
    :param origin: the optional origin snapshot name if the stream is for a clone.
    :type origin: bytes or None
    :param props: the properties to set on the snapshot as *received* properties.
    :type props: dict of bytes : Any

    :raises IOError: if an input / output error occurs while reading from the ``fd``.
    :raises DatasetExists: if the snapshot named ``snapname`` already exists.
    :raises DatasetExists: if the stream is a full stream and the destination filesystem already exists.
    :raises DatasetExists: if ``force`` is `True` but the destination filesystem could not
                           be rolled back to a matching snapshot because a newer snapshot
                           exists and it is an origin of a cloned filesystem.
    :raises StreamMismatch: if an incremental stream is received and the latest
                            snapshot of the destination filesystem does not match
                            the source snapshot of the stream.
    :raises StreamMismatch: if a full stream is received and the destination
                            filesystem already exists and it has at least one snapshot,
                            and ``force`` is `False`.
    :raises StreamMismatch: if an incremental clone stream is received but the specified
                            ``origin`` is not the actual received origin.
    :raises DestinationModified: if an incremental stream is received and the destination
                                 filesystem has been modified since the last snapshot
                                 and ``force`` is `False`.
    :raises DestinationModified: if a full stream is received and the destination
                                 filesystem already exists and it does not have any
                                 snapshots, and ``force`` is `False`.
    :raises DatasetNotFound: if the destination filesystem and its parent do not exist.
    :raises DatasetNotFound: if the ``origin`` is not `None` and does not exist.
    :raises DatasetBusy: if ``force`` is `True` but the destination filesystem could not
                         be rolled back to a matching snapshot because a newer snapshot
                         is held and could not be destroyed.
    :raises DatasetBusy: if another receive operation is being performed on the
                         destination filesystem.
    :raises BadStream: if the stream is corrupt or it is not recognized or it is
                       a compound stream or it is a clone stream, but ``origin``
                       is `None`.
    :raises BadStream: if a clone stream is received and the destination filesystem
                       already exists.
    :raises StreamFeatureNotSupported: if the stream has a feature that is not
                                       supported on this side.
    :raises PropertyInvalid: if one or more of the specified properties is invalid
                             or has an invalid type or value.
    :raises NameInvalid: if the name of either snapshot is invalid.
    :raises NameTooLong: if the name of either snapshot is too long.

    .. note::
        The ``origin`` is ignored if the actual stream is an incremental stream
        that is not a clone stream and the destination filesystem exists.
        If the stream is a full stream and the destination filesystem does not
        exist then the ``origin`` is checked for existence: if it does not exist
        :exc:`.DatasetNotFound` is raised, otherwise :exc:`.StreamMismatch` is
        raised, because that snapshot can not have any relation to the stream.

    .. note::
        If ``force`` is `True` and the stream is incremental then the destination
        filesystem is rolled back to a matching source snapshot if necessary.
        Intermediate snapshots are destroyed in that case.

        However, none of the existing snapshots may have the same name as
        ``snapname`` even if such a snapshot were to be destroyed.
        The existing ``snapname`` snapshot always causes :exc:`.SnapshotExists`
        to be raised.

        If ``force`` is `True` and the stream is a full stream then the destination
        filesystem is replaced with the received filesystem unless the former
        has any snapshots.  This prevents the destination filesystem from being
        rolled back / replaced.

    .. note::
        This interface does not work on dedup'd streams
        (those with ``DMU_BACKUP_FEATURE_DEDUP``).

    .. note::
        ``lzc_receive`` does not return until all of the stream is read from ``fd``
        and applied to the pool.

    .. note::
        ``lzc_receive`` does *not* close ``fd`` upon returning.
    """
    if origin is not None:
        c_origin = origin
    else:
        c_origin = _ffi.NULL
    if props is None:
        props = {}
    nvlist = nvlist_in(props)
    ret = _lib.lzc_receive(snapname, nvlist, c_origin, force, fd)
    errors.lzc_receive_translate_error(ret, snapname, fd, force, origin, props)