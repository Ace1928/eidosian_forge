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
def device_gradient_fn(inner_tapes, **gradient_kwargs):
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in inner_tapes))
    return cached_gradient_fn(numpy_tapes, **gradient_kwargs)