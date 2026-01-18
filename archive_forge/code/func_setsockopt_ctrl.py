import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
def setsockopt_ctrl(self, opt, value):
    """Enqueue setsockopt(opt, value) for ctrl_socket

        See zmq.Socket.setsockopt for details.
        """
    self._ctrl_sockopts.append((opt, value))