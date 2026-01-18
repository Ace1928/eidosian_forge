from typing import List, Union, Sequence, Dict, Optional, TYPE_CHECKING
from cirq import circuits, ops, value
from cirq.ops import Qid
from cirq._doc import document
def random_one_qubit_gate():
    return ops.PhasedXPowGate(phase_exponent=prng.rand(), exponent=prng.rand())