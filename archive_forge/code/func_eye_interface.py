import itertools
from functools import reduce
from typing import Generator, Iterable, Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import pennylane as qml
from pennylane.wires import Wires
def eye_interface(dim):
    if interface == 'scipy':
        return eye(2 ** dim, format='coo')
    return qml.math.cast_like(qml.math.eye(2 ** dim, like=interface), mat)