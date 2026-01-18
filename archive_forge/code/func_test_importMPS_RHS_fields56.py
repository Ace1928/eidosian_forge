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
def test_importMPS_RHS_fields56(self):
    """Import MPS file with RHS definitions in fields 5 & 6."""
    with tempfile.NamedTemporaryFile(delete=False) as h:
        h.write(str.encode(EXAMPLE_MPS_RHS56))
    _, problem = LpProblem.fromMPS(h.name)
    os.unlink(h.name)
    self.assertEqual(problem.constraints['LIM2'].constant, -10)