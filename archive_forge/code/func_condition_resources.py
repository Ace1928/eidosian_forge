from __future__ import annotations
import dataclasses
from typing import Iterable, Tuple, Set, Union, TypeVar, TYPE_CHECKING
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.register import Register
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.quantumregister import QuantumRegister
def condition_resources(condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr) -> LegacyResources:
    """Get the legacy classical resources (:class:`.Clbit` and :class:`.ClassicalRegister`)
    referenced by a legacy condition or an :class:`~.expr.Expr`."""
    if isinstance(condition, expr.Expr):
        return node_resources(condition)
    target, _ = condition
    if isinstance(target, ClassicalRegister):
        return LegacyResources(tuple(target), (target,))
    return LegacyResources((target,), ())