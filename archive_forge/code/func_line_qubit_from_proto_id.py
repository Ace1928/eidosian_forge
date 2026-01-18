import re
from typing import TYPE_CHECKING
import cirq
def line_qubit_from_proto_id(proto_id: str) -> cirq.LineQubit:
    """Parse a proto id to a `cirq.LineQubit`.

    Proto ids for line qubits are integer strings representing the `x`
    attribute of the line qubit.

    Args:
        proto_id: The id to convert.

    Returns:
        A `cirq.LineQubit` corresponding to the proto id.

    Raises:
        ValueError: If the string is not an integer.
    """
    try:
        return cirq.LineQubit(x=int(proto_id))
    except ValueError:
        raise ValueError(f'Line qubit proto id must be an int but was {proto_id}')