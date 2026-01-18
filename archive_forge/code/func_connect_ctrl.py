import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
def connect_ctrl(self, addr):
    """Enqueue ZMQ address for connecting on ctrl_socket.

        See zmq.Socket.connect for details.
        """
    self._ctrl_connects.append(addr)