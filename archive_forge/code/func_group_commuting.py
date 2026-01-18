from __future__ import annotations
from typing import TYPE_CHECKING, List
from collections import defaultdict
from collections.abc import Mapping, Sequence, Iterable
from numbers import Number
from copy import deepcopy
import numpy as np
import rustworkx as rx
from qiskit._accelerate.sparse_pauli_op import unordered_unique, decompose_dense
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
def group_commuting(self, qubit_wise: bool=False) -> list[SparsePauliOp]:
    """Partition a SparsePauliOp into sets of commuting Pauli strings.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.  For example:

                .. code-block:: python

                    >>> op = SparsePauliOp.from_list([("XX", 2), ("YY", 1), ("IZ",2j), ("ZZ",1j)])
                    >>> op.group_commuting()
                    [SparsePauliOp(["IZ", "ZZ"], coeffs=[0.+2.j, 0.+1j]),
                     SparsePauliOp(["XX", "YY"], coeffs=[2.+0.j, 1.+0.j])]
                    >>> op.group_commuting(qubit_wise=True)
                    [SparsePauliOp(['XX'], coeffs=[2.+0.j]),
                     SparsePauliOp(['YY'], coeffs=[1.+0.j]),
                     SparsePauliOp(['IZ', 'ZZ'], coeffs=[0.+2.j, 0.+1.j])]

        Returns:
            list[SparsePauliOp]: List of SparsePauliOp where each SparsePauliOp contains
                commuting Pauli operators.
        """
    graph = self._create_graph(qubit_wise)
    coloring_dict = rx.graph_greedy_color(graph)
    groups = defaultdict(list)
    for idx, color in coloring_dict.items():
        groups[color].append(idx)
    return [self[group] for group in groups.values()]