import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, ops, protocols, value
from cirq.linalg import transformations
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements
def flip_inversion(op: 'cirq.Operation', _) -> 'cirq.OP_TREE':
    if isinstance(op.gate, ops.MeasurementGate):
        return [ops.X(q) if b else ops.I(q) for q, b in zip(op.qubits, op.gate.full_invert_mask())]
    return op