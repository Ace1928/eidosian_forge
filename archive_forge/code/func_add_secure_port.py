import abc
import grpc
@abc.abstractmethod
def add_secure_port(self, address, server_credentials):
    """Reserves a port for secure RPC service after this Server becomes active.

        This method may only be called before calling this Server's start method is
        called.

        Args:
          address: The address for which to open a port.
          server_credentials: A ServerCredentials.

        Returns:
          An integer port on which RPCs will be serviced after this link has been
            started. This is typically the same number as the port number contained
            in the passed address, but will likely be different if the port number
            contained in the passed address was zero.
        """
    raise NotImplementedError()