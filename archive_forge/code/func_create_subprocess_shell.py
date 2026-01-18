import collections
import subprocess
from . import events
from . import futures
from . import protocols
from . import streams
from . import tasks
from .coroutines import coroutine
from .log import logger
@coroutine
def create_subprocess_shell(cmd, stdin=None, stdout=None, stderr=None, loop=None, limit=streams._DEFAULT_LIMIT, **kwds):
    if loop is None:
        loop = events.get_event_loop()
    protocol_factory = lambda: SubprocessStreamProtocol(limit=limit, loop=loop)
    transport, protocol = (yield from loop.subprocess_shell(protocol_factory, cmd, stdin=stdin, stdout=stdout, stderr=stderr, **kwds))
    return Process(transport, protocol, loop)