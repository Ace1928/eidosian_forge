from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def has_classical_cotransform(self) -> bool:
    """Check if the transform program has some classical cotransforms.

        Returns:
            bool: Boolean
        """
    return any((t.classical_cotransform is not None for t in self))