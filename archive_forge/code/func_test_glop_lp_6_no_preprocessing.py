import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_glop_lp_6_no_preprocessing(self) -> None:
    from ortools.glop import parameters_pb2
    params = parameters_pb2.GlopParameters()
    params.use_preprocessing = False
    StandardTestLPs.test_lp_6(solver='GLOP', parameters_proto=params)