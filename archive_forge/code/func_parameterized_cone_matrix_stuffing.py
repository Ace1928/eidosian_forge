import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def parameterized_cone_matrix_stuffing():
    ConeMatrixStuffing().apply(problem)