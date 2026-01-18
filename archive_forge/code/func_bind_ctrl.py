import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
def bind_ctrl(self, addr):
    """Enqueue ZMQ address for binding on ctrl_socket.

        See zmq.Socket.bind for details.
        """
    self._ctrl_binds.append(addr)