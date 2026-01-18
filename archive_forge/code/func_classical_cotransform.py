import functools
import os
import copy
import warnings
import types
from typing import Sequence
import pennylane as qml
from pennylane.typing import ResultBatch
@property
def classical_cotransform(self):
    """The stored quantum transform's classical co-transform."""
    return self._classical_cotransform