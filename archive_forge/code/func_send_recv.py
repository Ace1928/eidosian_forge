import abc
from cupyx.distributed import _store
@abc.abstractmethod
def send_recv(self, in_array, out_array, peer, stream=None):
    pass