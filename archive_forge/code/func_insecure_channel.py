import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def insecure_channel(host, port):
    """Creates an insecure Channel to a remote host.

    Args:
      host: The name of the remote host to which to connect.
      port: The port of the remote host to which to connect.
        If None only the 'host' part will be used.

    Returns:
      A Channel to the remote host through which RPCs may be conducted.
    """
    channel = grpc.insecure_channel(host if port is None else '%s:%d' % (host, port))
    return Channel(channel)