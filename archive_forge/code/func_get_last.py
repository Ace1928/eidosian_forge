from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def get_last(self):
    """Get the last transform container.

        Returns:
            TransformContainer: The last transform in the program.

        Raises:
            TransformError: It raises an error if the program is empty.
        """
    if self:
        return self._transform_program[-1]
    raise TransformError('The transform program is empty and you cannot get the last transform container.')