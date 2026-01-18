import contextlib
import copy
from collections import Counter
from typing import List, Union, Optional, Sequence
import pennylane as qml
from pennylane.measurements import (
from pennylane.typing import TensorLike
from pennylane.operation import Observable, Operator, Operation, _UNSET_BATCH_SIZE
from pennylane.pytrees import register_pytree
from pennylane.queuing import AnnotatedQueue, process_queue
from pennylane.wires import Wires
@property
def numeric_type(self):
    """Returns the expected numeric type of the quantum script result by inspecting
        its measurements.

        Returns:
            Union[type, Tuple[type]]: The numeric type corresponding to the result type of the
            quantum script, or a tuple of such types if the script contains multiple measurements

        **Example:**

        .. code-block:: pycon

            >>> dev = qml.device('default.qubit', wires=2)
            >>> qs = QuantumScript(measurements=[qml.state()])
            >>> qs.numeric_type
            complex
        """
    types = tuple((observable.numeric_type for observable in self.measurements))
    return types[0] if len(types) == 1 else types