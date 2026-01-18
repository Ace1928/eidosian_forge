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
def test_importMPS_integer(self):
    name = self._testMethodName
    prob = LpProblem(name, const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, None, const.LpInteger)
    prob += (1.1 * x + 4.1 * y + 9.1 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7.5, 'c3')
    filename = name + '.mps'
    prob.writeMPS(filename)
    _vars, prob2 = LpProblem.fromMPS(filename, sense=prob.sense)
    _dict1 = getSortedDict(prob)
    _dict2 = getSortedDict(prob2)
    print('\t Testing reading MPS files - integer variable')
    self.assertDictEqual(_dict1, _dict2)