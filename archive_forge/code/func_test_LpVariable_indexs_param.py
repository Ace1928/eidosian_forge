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
def test_LpVariable_indexs_param(self):
    """
            Test that 'indexs' param continues to work
            """
    prob = LpProblem(self._testMethodName, const.LpMinimize)
    customers = [1, 2, 3]
    agents = ['A', 'B', 'C']
    print("\t Testing 'indexs' param continues to work for LpVariable.dicts")
    assign_vars = LpVariable.dicts(name='test', indices=(customers, agents))
    for k, v in assign_vars.items():
        for a, b in v.items():
            self.assertIsInstance(b, LpVariable)
    assign_vars = LpVariable.dicts('test', (customers, agents))
    for k, v in assign_vars.items():
        for a, b in v.items():
            self.assertIsInstance(b, LpVariable)
    print("\t Testing 'indexs' param continues to work for LpVariable.matrix")
    assign_vars_matrix = LpVariable.matrix(name='test', indices=(customers, agents))
    for a in assign_vars_matrix:
        for b in a:
            self.assertIsInstance(b, LpVariable)
    assign_vars_matrix = LpVariable.matrix('test', (customers, agents))
    for a in assign_vars_matrix:
        for b in a:
            self.assertIsInstance(b, LpVariable)