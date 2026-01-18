import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
def test_equalityErrors(self):
    m = self.m
    e = m.a == m.b
    with self.assertRaisesRegex(TypeError, 'Attempting to use a non-numeric type \\(EqualityExpression\\) in a numeric expression context.'):
        e == m.a
    with self.assertRaisesRegex(TypeError, 'Attempting to use a non-numeric type \\(EqualityExpression\\) in a numeric expression context.'):
        m.a == e
    with self.assertRaisesRegex(TypeError, 'Argument .* is an indexed numeric value'):
        m.x == m.a
    with self.assertRaisesRegex(TypeError, 'Argument .* is an indexed numeric value'):
        m.a == m.x