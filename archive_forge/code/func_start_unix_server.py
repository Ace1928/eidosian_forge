import socket
from . import coroutines
from . import events
from . import futures
from . import protocols
from .coroutines import coroutine
from .log import logger
@coroutine
def start_unix_server(client_connected_cb, path=None, *, loop=None, limit=_DEFAULT_LIMIT, **kwds):
    """Similar to `start_server` but works with UNIX Domain Sockets."""
    if loop is None:
        loop = events.get_event_loop()

    def factory():
        reader = StreamReader(limit=limit, loop=loop)
        protocol = StreamReaderProtocol(reader, client_connected_cb, loop=loop)
        return protocol
    return (yield from loop.create_unix_server(factory, path, **kwds))