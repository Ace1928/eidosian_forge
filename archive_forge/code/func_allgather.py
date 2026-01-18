from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import (
@abstractmethod
def allgather(self, tensor_list, tensor, allgather_options=AllGatherOptions()):
    raise NotImplementedError()