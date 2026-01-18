import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
def default_qnode_transform(self, qnode, targs, tkwargs):
    """
        The default method that takes in a QNode and returns another QNode
        with the transform applied.
        """
    qnode = copy.copy(qnode)
    if self.expand_transform:
        qnode.add_transform(TransformContainer(self._expand_transform, targs, tkwargs, use_argnum=self._use_argnum_in_expand))
    qnode.add_transform(TransformContainer(self._transform, targs, tkwargs, self._classical_cotransform, self._is_informative, self._final_transform))
    return qnode