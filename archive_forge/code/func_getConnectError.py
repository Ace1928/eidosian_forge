import socket
from incremental import Version
from twisted.python import deprecate
def getConnectError(e):
    """Given a socket exception, return connection error."""
    if isinstance(e, Exception):
        args = e.args
    else:
        args = e
    try:
        number, string = args
    except ValueError:
        return ConnectError(string=e)
    if hasattr(socket, 'gaierror') and isinstance(e, socket.gaierror):
        klass = UnknownHostError
    else:
        klass = errnoMapping.get(number, ConnectError)
    return klass(number, string)