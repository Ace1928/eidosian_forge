from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, apply_gate_to_lth_target, arithmetic_gates
from cirq_ft.algos import prepare_uniform_superposition as prep_u
from cirq_ft.algos import (
def get_walk_operator_for_hubbard_model(x_dim: int, y_dim: int, t: int, mu: int) -> 'qubitization_walk_operator.QubitizationWalkOperator':
    select = SelectHubbard(x_dim, y_dim)
    prepare = PrepareHubbard(x_dim, y_dim, t, mu)
    return qubitization_walk_operator.QubitizationWalkOperator(select=select, prepare=prepare)