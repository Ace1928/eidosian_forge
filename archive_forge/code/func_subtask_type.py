from abc import ABCMeta, abstractmethod
from collections.abc import Callable
@property
@abstractmethod
def subtask_type(self):
    pass