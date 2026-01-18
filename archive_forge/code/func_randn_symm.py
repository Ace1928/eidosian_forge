import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def randn_symm(n):
    A = np.random.randn(n, n)
    return (A + A.T) / 2