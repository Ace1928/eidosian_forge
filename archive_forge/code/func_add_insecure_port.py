import abc
import grpc
@abc.abstractmethod
def add_insecure_port(self, address):
    """Reserves a port for insecure RPC service once this Server becomes active.

        This method may only be called before calling this Server's start method is
        called.

        Args:
          address: The address for which to open a port.

        Returns:
          An integer port on which RPCs will be serviced after this link has been
            started. This is typically the same number as the port number contained
            in the passed address, but will likely be different if the port number
            contained in the passed address was zero.
        """
    raise NotImplementedError()