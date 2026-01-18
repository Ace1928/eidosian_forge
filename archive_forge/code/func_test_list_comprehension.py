import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
def test_list_comprehension(self):
    expr = Expression('[n for n in numbers if n < 2]')
    self.assertEqual([0, 1], expr.evaluate({'numbers': range(5)}))
    expr = Expression('[(i, n + 1) for i, n in enumerate(numbers)]')
    self.assertEqual([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], expr.evaluate({'numbers': range(5)}))
    expr = Expression('[offset + n for n in numbers]')
    self.assertEqual([2, 3, 4, 5, 6], expr.evaluate({'numbers': range(5), 'offset': 2}))
    expr = Expression('[n for group in groups for n in group]')
    self.assertEqual([0, 1, 0, 1, 2], expr.evaluate({'groups': [range(2), range(3)]}))
    expr = Expression('[(a, b) for a in x for b in y]')
    self.assertEqual([('x0', 'y0'), ('x0', 'y1'), ('x1', 'y0'), ('x1', 'y1')], expr.evaluate({'x': ['x0', 'x1'], 'y': ['y0', 'y1']}))