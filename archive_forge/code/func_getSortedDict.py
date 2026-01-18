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
def getSortedDict(prob, keyCons='name', keyVars='name'):
    _dict = prob.toDict()
    _dict['constraints'].sort(key=lambda v: v[keyCons])
    _dict['variables'].sort(key=lambda v: v[keyVars])
    return _dict