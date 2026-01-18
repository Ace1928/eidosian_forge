import sys
import _overlapped
import _winapi
import errno
import math
import msvcrt
import socket
import struct
import time
import weakref
from . import events
from . import base_subprocess
from . import futures
from . import exceptions
from . import proactor_events
from . import selector_events
from . import tasks
from . import windows_utils
from .log import logger
def loop_accept_pipe(f=None):
    pipe = None
    try:
        if f:
            pipe = f.result()
            server._free_instances.discard(pipe)
            if server.closed():
                pipe.close()
                return
            protocol = protocol_factory()
            self._make_duplex_pipe_transport(pipe, protocol, extra={'addr': address})
        pipe = server._get_unconnected_pipe()
        if pipe is None:
            return
        f = self._proactor.accept_pipe(pipe)
    except BrokenPipeError:
        if pipe and pipe.fileno() != -1:
            pipe.close()
        self.call_soon(loop_accept_pipe)
    except OSError as exc:
        if pipe and pipe.fileno() != -1:
            self.call_exception_handler({'message': 'Pipe accept failed', 'exception': exc, 'pipe': pipe})
            pipe.close()
        elif self._debug:
            logger.warning('Accept pipe failed on pipe %r', pipe, exc_info=True)
        self.call_soon(loop_accept_pipe)
    except exceptions.CancelledError:
        if pipe:
            pipe.close()
    else:
        server._accept_pipe_future = f
        f.add_done_callback(loop_accept_pipe)