import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice
def connect_mon(self, addr):
    """Enqueue ZMQ address for connecting on mon_socket.

        See zmq.Socket.connect for details.
        """
    self._mon_connects.append(addr)