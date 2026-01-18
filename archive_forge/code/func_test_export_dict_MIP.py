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
def test_export_dict_MIP(self):
    import copy
    prob = LpProblem('test_export_dict_MIP', const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0, None, const.LpInteger)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7.5, 'c3')
    data = prob.toDict()
    data_backup = copy.deepcopy(data)
    var1, prob1 = LpProblem.fromDict(data)
    x, y, z = (var1[name] for name in ['x', 'y', 'z'])
    print('\t Testing export dict MIP')
    pulpTestCheck(prob1, self.solver, [const.LpStatusOptimal], {x: 3, y: -0.5, z: 7})
    self.assertDictEqual(data, data_backup)