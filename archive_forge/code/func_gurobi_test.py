import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
def gurobi_test(test_item):

    @functools.wraps(test_item)
    def skip_wrapper(*args, **kwargs):
        if gp is None:
            raise unittest.SkipTest("No gurobipy, can't check license")
        try:
            test_item(*args, **kwargs)
        except gp.GurobiError as ge:
            if ge.errno == gp.GRB.Error.SIZE_LIMIT_EXCEEDED:
                raise unittest.SkipTest('Size-limited Gurobi license')
            if ge.errno == gp.GRB.Error.NO_LICENSE:
                raise unittest.SkipTest('No Gurobi license')
            raise
    return skip_wrapper