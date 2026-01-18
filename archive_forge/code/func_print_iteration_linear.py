from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def print_iteration_linear(iteration, cost, cost_reduction, step_norm, optimality):
    if cost_reduction is None:
        cost_reduction = ' ' * 15
    else:
        cost_reduction = f'{cost_reduction:^15.2e}'
    if step_norm is None:
        step_norm = ' ' * 15
    else:
        step_norm = f'{step_norm:^15.2e}'
    print(f'{iteration:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}')