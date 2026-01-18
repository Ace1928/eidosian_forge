from cmath import exp
from math import cos, sin, sqrt
import pytest
import numpy as np
from scipy.linalg import block_diag
from flaky import flaky
import pennylane as qml
def adjoint_tuple(op, orig_mat):
    """Returns op constructor and matrix for provided base ops."""
    mat = qml.math.conj(qml.math.transpose(orig_mat))
    return (qml.adjoint(op), mat)