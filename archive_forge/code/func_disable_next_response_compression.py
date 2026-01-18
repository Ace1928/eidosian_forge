import abc
import grpc
@abc.abstractmethod
def disable_next_response_compression(self):
    """Disables compression of the next response passed by the application."""
    raise NotImplementedError()