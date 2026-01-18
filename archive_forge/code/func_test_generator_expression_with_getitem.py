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
def test_generator_expression_with_getitem(self):
    items = [{'name': 'a', 'value': 1}, {'name': 'b', 'value': 2}]
    expr = Expression("list(i['name'] for i in items if i['value'] > 1)")
    self.assertEqual(['b'], expr.evaluate({'items': items}))