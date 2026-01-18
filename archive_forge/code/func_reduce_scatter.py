import abc
from cupyx.distributed import _store
@abc.abstractmethod
def reduce_scatter(self, in_array, out_array, count, op='sum', stream=None):
    pass