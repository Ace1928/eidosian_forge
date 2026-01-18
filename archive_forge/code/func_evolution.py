from functools import singledispatch
from pennylane.operation import Operator
from pennylane.ops import Evolution
from pennylane.pulse import ParametrizedEvolution, ParametrizedHamiltonian
@evolve.register
def evolution(op: Operator, coeff: float=1, num_steps: int=None):
    return Evolution(op, coeff, num_steps)