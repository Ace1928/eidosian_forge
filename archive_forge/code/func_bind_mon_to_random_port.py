import zmq
from zmq.devices.basedevice import Device, ProcessDevice, ThreadDevice
def bind_mon_to_random_port(self, addr, *args, **kwargs):
    """Enqueue a random port on the given interface for binding on
        mon_socket.

        See zmq.Socket.bind_to_random_port for details.

        .. versionadded:: 18.0
        """
    port = self._reserve_random_port(addr, *args, **kwargs)
    self.bind_mon('%s:%i' % (addr, port))
    return port