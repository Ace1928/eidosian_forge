from itertools import combinations
import numpy as np
import pennylane as qml
def format_nvec(nvec):
    """Nice strings representing tuples of integers."""
    if isinstance(nvec, int):
        return str(nvec)
    return ' '.join((f'{n: }' for n in nvec))