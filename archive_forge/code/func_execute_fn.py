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
def execute_fn(internal_tapes) -> Tuple[ResultBatch, Tuple]:
    """A wrapper around device.execute that adds an empty tuple instead of derivatives.

                Closure Variables:
                    device: the device to execute on
                    config: the ExecutionConfig that specifies how to perform the simulations.
                """
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in internal_tapes))
    return (device.execute(numpy_tapes, config), tuple())