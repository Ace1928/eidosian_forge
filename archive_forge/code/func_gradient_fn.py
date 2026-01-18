import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple
import logging
from cachetools import LRUCache, Cache
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch
from .set_shots import set_shots
from .jacobian_products import (
def gradient_fn(internal_tapes):
    """A partial function that wraps compute_derivatives method of the device.

                Closure Variables:
                    device: the device to execute on
                    config: the ExecutionConfig that specifies how to take the derivative.
                """
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes))
    return device.compute_derivatives(numpy_tapes, config)