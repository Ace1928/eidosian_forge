from collections import defaultdict
from typing import (
import numbers
import numpy as np
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from scipy.sparse import csr_matrix
from cirq import linalg, protocols, qis, value
from cirq._doc import document
from cirq.linalg import operator_spaces
from cirq.ops import identity, raw_types, pauli_gates, pauli_string
from cirq.ops.pauli_string import PauliString, _validate_qubit_mapping
from cirq.ops.projector import ProjectorString
from cirq.value.linear_dict import _format_terms
def key_json(k: UnitPauliStringT):
    return [list(e) for e in sorted(k)]