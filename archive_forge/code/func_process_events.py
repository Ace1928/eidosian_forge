from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
def process_events(self, received: EventResult) -> None:
    for i in range(received):
        entry = self._events[i]
        if entry.lpCompletionKey == CKeys.AFD_POLL:
            lpo = entry.lpOverlapped
            op = self._afd_ops.pop(lpo)
            waiters = op.waiters
            if waiters.current_op is not op:
                pass
            else:
                waiters.current_op = None
                if lpo.Internal != 0:
                    code = ntdll.RtlNtStatusToDosError(lpo.Internal)
                    raise_winerror(code)
                flags = op.poll_info.Handles[0].Events
                if waiters.read_task and flags & READABLE_FLAGS:
                    _core.reschedule(waiters.read_task)
                    waiters.read_task = None
                if waiters.write_task and flags & WRITABLE_FLAGS:
                    _core.reschedule(waiters.write_task)
                    waiters.write_task = None
                self._refresh_afd(op.poll_info.Handles[0].Handle)
        elif entry.lpCompletionKey == CKeys.WAIT_OVERLAPPED:
            waiter = self._overlapped_waiters.pop(entry.lpOverlapped)
            overlapped = entry.lpOverlapped
            transferred = entry.dwNumberOfBytesTransferred
            info = CompletionKeyEventInfo(lpOverlapped=overlapped, dwNumberOfBytesTransferred=transferred)
            _core.reschedule(waiter, Value(info))
        elif entry.lpCompletionKey == CKeys.LATE_CANCEL:
            self._posted_too_late_to_cancel.remove(entry.lpOverlapped)
            try:
                waiter = self._overlapped_waiters.pop(entry.lpOverlapped)
            except KeyError:
                pass
            else:
                exc = _core.TrioInternalError(f"Failed to cancel overlapped I/O in {waiter.name} and didn't receive the completion either. Did you forget to call register_with_iocp()?")
                raise exc
        elif entry.lpCompletionKey == CKeys.FORCE_WAKEUP:
            pass
        else:
            queue = self._completion_key_queues[entry.lpCompletionKey]
            overlapped = int(ffi.cast('uintptr_t', entry.lpOverlapped))
            transferred = entry.dwNumberOfBytesTransferred
            info = CompletionKeyEventInfo(lpOverlapped=overlapped, dwNumberOfBytesTransferred=transferred)
            queue.put_nowait(info)