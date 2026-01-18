import itertools
from typing import Sequence, Tuple
import cirq
import cirq_ft
import pytest
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.infra.bit_tools import iter_bits
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def nth_operation(self, context: cirq.DecompositionContext, control: cirq.Qid, i: int, j: int, k: int, t1: Sequence[cirq.Qid], t2: Sequence[cirq.Qid], t3: Sequence[cirq.Qid]) -> cirq.OP_TREE:
    yield [cirq.CNOT(control, t1[i]), cirq.CNOT(control, t2[j]), cirq.CNOT(control, t3[k])]