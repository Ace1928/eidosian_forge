from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def classical_preprocessing(program, *args, **kwargs):
    """Returns the trainable gate parameters for a given QNode input."""
    kwargs.pop('shots', None)
    kwargs.pop('argnums', None)
    qnode.construct(args, kwargs)
    tape = qnode.qtape
    tapes, _ = program((tape,))
    res = tuple((qml.math.stack(tape.get_parameters(trainable_only=True)) for tape in tapes))
    if len(tapes) == 1:
        return res[0]
    return res