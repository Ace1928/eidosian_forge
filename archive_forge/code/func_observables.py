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
def observables(self) -> List[Union[MeasurementProcess, Observable]]:
    """Returns the observables on the quantum script.

        Returns:
            list[.MeasurementProcess, .Observable]]: list of observables

        **Example**

        >>> ops = [qml.StatePrep([0, 1], 0), qml.RX(0.432, 0)]
        >>> qscript = QuantumScript(ops, [qml.expval(qml.Z(0))])
        >>> qscript.observables
        [expval(Z(0))]
        """
    obs = []
    for m in self.measurements:
        if m.obs is not None:
            obs.append(m.obs)
        else:
            obs.append(m)
    return obs