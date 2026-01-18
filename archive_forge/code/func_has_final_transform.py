from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
@property
def has_final_transform(self) -> bool:
    """``True`` if the transform program has a terminal transform."""
    return self[-1].final_transform if self else False