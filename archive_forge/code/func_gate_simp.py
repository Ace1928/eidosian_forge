from itertools import chain
import random
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.iterables import is_sequence
def gate_simp(circuit):
    """Simplifies gates symbolically

    It first sorts gates using gate_sort. It then applies basic
    simplification rules to the circuit, e.g., XGate**2 = Identity
    """
    circuit = gate_sort(circuit)
    if isinstance(circuit, Add):
        return sum((gate_simp(t) for t in circuit.args))
    elif isinstance(circuit, Mul):
        circuit_args = circuit.args
    elif isinstance(circuit, Pow):
        b, e = circuit.as_base_exp()
        circuit_args = (gate_simp(b) ** e,)
    else:
        return circuit
    for i in range(len(circuit_args)):
        if isinstance(circuit_args[i], Pow):
            if isinstance(circuit_args[i].base, (HadamardGate, XGate, YGate, ZGate)) and isinstance(circuit_args[i].exp, Number):
                newargs = circuit_args[:i] + (circuit_args[i].base ** (circuit_args[i].exp % 2),) + circuit_args[i + 1:]
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, PhaseGate):
                newargs = circuit_args[:i]
                newargs = newargs + (ZGate(circuit_args[i].base.args[0]) ** Integer(circuit_args[i].exp / 2), circuit_args[i].base ** (circuit_args[i].exp % 2))
                newargs = newargs + circuit_args[i + 1:]
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, TGate):
                newargs = circuit_args[:i]
                newargs = newargs + (PhaseGate(circuit_args[i].base.args[0]) ** Integer(circuit_args[i].exp / 2), circuit_args[i].base ** (circuit_args[i].exp % 2))
                newargs = newargs + circuit_args[i + 1:]
                circuit = gate_simp(Mul(*newargs))
                break
    return circuit