import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self
from cirq import ops, protocols
@staticmethod
def for_gate(val: Any, start: int=0, step: int=1) -> List['LineQid']:
    """Returns a range of line qids with the same qid shape as the gate.

        Args:
            val: Any value that supports the `cirq.qid_shape` protocol.  Usually
                a gate.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
    from cirq.protocols.qid_shape_protocol import qid_shape
    return LineQid.for_qid_shape(qid_shape(val), start=start, step=step)