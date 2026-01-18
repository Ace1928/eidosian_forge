from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_getArgNames(self):
    """
        Type annotations should be included in the set of
        """
    spec = ArgSpec(args=('a', 'b'), varargs=None, varkw=None, defaults=None, kwonlyargs=(), kwonlydefaults=None, annotations=(('a', int), ('b', str)))
    self.assertEqual(_getArgNames(spec), {'a', 'b', ('a', int), ('b', str)})