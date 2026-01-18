from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def pop_front(self):
    """Pop the transform container at the beginning of the program.

        Returns:
            TransformContainer: The transform container at the beginning of the program.
        """
    return self._transform_program.pop(0)