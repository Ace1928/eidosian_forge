from math import copysign
import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator
def print_header_linear():
    print('{:^15}{:^15}{:^15}{:^15}{:^15}'.format('Iteration', 'Cost', 'Cost reduction', 'Step norm', 'Optimality'))