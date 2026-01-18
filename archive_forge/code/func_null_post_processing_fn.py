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
def null_post_processing_fn(results):
    """A null post processing function used because the user requested not to use the device batch transform."""
    return results