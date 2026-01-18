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
def test_makeDict_default_value(self):
    """
            Test if makeDict is returning a default value when specified.
            """
    headers = [['A', 'B'], ['C', 'D']]
    values = [[1, 2], [3, 4]]
    dict_with_default = makeDict(headers, values, default=0)
    dict_without_default = makeDict(headers, values)
    print('\t Testing makeDict default value behavior')
    self.assertEqual(dict_with_default['X']['Y'], 0)
    _func = lambda: dict_without_default['X']['Y']
    self.assertRaises(KeyError, _func)