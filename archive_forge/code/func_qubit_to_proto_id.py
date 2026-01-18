import re
from typing import TYPE_CHECKING
import cirq
def qubit_to_proto_id(q: cirq.Qid) -> str:
    """Return a proto id for a `cirq.Qid`.

    For `cirq.GridQubit`s this id `{row}_{col}` where `{row}` is the integer
    row of the grid qubit, and `{col}` is the integer column of the qubit.

    For `cirq.NamedQubit`s this id is the name.

    For `cirq.LineQubit`s this is string of the `x` attribute.
    """
    if isinstance(q, cirq.GridQubit):
        return f'{q.row}_{q.col}'
    elif isinstance(q, cirq.NamedQubit):
        return q.name
    elif isinstance(q, cirq.LineQubit):
        return f'{q.x}'
    else:
        raise ValueError(f'Qubits of type {type(q)} do not support proto id')