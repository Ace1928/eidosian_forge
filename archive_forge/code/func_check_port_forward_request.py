import threading
from paramiko import util
from paramiko.common import (
def check_port_forward_request(self, address, port):
    """
        Handle a request for port forwarding.  The client is asking that
        connections to the given address and port be forwarded back across
        this ssh connection.  An address of ``"0.0.0.0"`` indicates a global
        address (any address associated with this server) and a port of ``0``
        indicates that no specific port is requested (usually the OS will pick
        a port).

        The default implementation always returns ``False``, rejecting the
        port forwarding request.  If the request is accepted, you should return
        the port opened for listening.

        :param str address: the requested address
        :param int port: the requested port
        :return:
            the port number (`int`) that was opened for listening, or ``False``
            to reject
        """
    return False