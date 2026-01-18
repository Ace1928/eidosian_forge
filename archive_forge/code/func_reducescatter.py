from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import (
@abstractmethod
def reducescatter(self, tensor, tensor_list, reducescatter_options=ReduceScatterOptions()):
    raise NotImplementedError()