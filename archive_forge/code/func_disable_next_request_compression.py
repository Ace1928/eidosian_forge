import abc
import grpc
@abc.abstractmethod
def disable_next_request_compression(self):
    """Disables compression of the next request passed by the application."""
    raise NotImplementedError()