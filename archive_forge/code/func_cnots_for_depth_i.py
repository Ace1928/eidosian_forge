from typing import Tuple
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
def cnots_for_depth_i(i: int, q: NDArray[cirq.Qid]) -> cirq.OP_TREE:
    for c, t in zip(q[:2 ** i], q[2 ** i:min(len(q), 2 ** (i + 1))]):
        yield cirq.CNOT(c, t)