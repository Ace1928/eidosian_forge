import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
def execute_and_compute_jacobian(self, tapes: Batch) -> Tuple:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('execute_and_compute_jacobian called with %s', tapes)
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
    return self._device.execute_and_compute_derivatives(numpy_tapes, self._execution_config)