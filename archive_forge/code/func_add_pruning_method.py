import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def add_pruning_method(self, method):
    """Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
    if not isinstance(method, BasePruningMethod) and method is not None:
        raise TypeError(f'{type(method)} is not a BasePruningMethod subclass')
    elif method is not None and self._tensor_name != method._tensor_name:
        raise ValueError(f"Can only add pruning methods acting on the parameter named '{self._tensor_name}' to PruningContainer {self}." + f" Found '{method._tensor_name}'")
    self._pruning_methods += (method,)