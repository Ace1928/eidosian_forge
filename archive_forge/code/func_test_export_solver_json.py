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
def test_export_solver_json(self):
    name = self._testMethodName
    prob = LpProblem(name, const.LpMinimize)
    x = LpVariable('x', 0, 4)
    y = LpVariable('y', -1, 1)
    z = LpVariable('z', 0)
    w = LpVariable('w', 0)
    prob += (x + 4 * y + 9 * z, 'obj')
    prob += (x + y <= 5, 'c1')
    prob += (x + z >= 10, 'c2')
    prob += (-y + z == 7, 'c3')
    prob += (w >= 0, 'c4')
    self.solver.mip = True
    logFilename = name + '.log'
    if self.solver.name == 'CPLEX_CMD':
        self.solver.optionsDict = dict(gapRel=0.1, gapAbs=1, maxMemory=1000, maxNodes=1, threads=1, logPath=logFilename, warmStart=True)
    elif self.solver.name in ['GUROBI_CMD', 'COIN_CMD', 'PULP_CBC_CMD']:
        self.solver.optionsDict = dict(gapRel=0.1, gapAbs=1, threads=1, logPath=logFilename, warmStart=True)
    filename = name + '.json'
    self.solver.toJson(filename, indent=4)
    solver1 = getSolverFromJson(filename)
    try:
        os.remove(filename)
    except:
        pass
    print('\t Testing continuous LP solution - export solver JSON')
    pulpTestCheck(prob, solver1, [const.LpStatusOptimal], {x: 4, y: -1, z: 6, w: 0})