import itertools
from typing import Callable, Sequence, Tuple
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import unary_iteration_gate
@classmethod
def make_on(cls, *, nth_gate: Callable[..., cirq.Gate], **quregs: Sequence[cirq.Qid]) -> cirq.Operation:
    """Helper constructor to automatically deduce bitsize attributes."""
    return ApplyGateToLthQubit(infra.SelectionRegister('selection', len(quregs['selection']), len(quregs['target'])), nth_gate=nth_gate, control_regs=infra.Register('control', len(quregs['control']))).on_registers(**quregs)