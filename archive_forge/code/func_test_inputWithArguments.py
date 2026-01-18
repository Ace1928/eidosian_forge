from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_inputWithArguments(self):
    """
        If an input takes an argument, it will pass that along to its output.
        """

    class Mechanism(object):
        m = MethodicalMachine()

        @m.input()
        def input(self, x, y=1):
            """an input"""

        @m.state(initial=True)
        def state(self):
            """a state"""

        @m.output()
        def output(self, x, y=1):
            self._x = x
            return x + y
        state.upon(input, state, [output])
    m = Mechanism()
    self.assertEqual(m.input(3), [4])
    self.assertEqual(m._x, 3)