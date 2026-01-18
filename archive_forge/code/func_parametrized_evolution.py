from functools import singledispatch
from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian
@evolve.register
def parametrized_evolution(op: ParametrizedHamiltonian, **kwargs):
    return ParametrizedEvolution(H=op, **kwargs)