from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_machineItselfIsPrivate(self):
    """
        L{MethodicalMachine} is an implementation detail.  If you attempt to
        access it on an instance of your class, you will get an exception.
        However, since tools may need to access it for the purposes of, for
        example, visualization, you may access it on the class itself.
        """
    expectedMachine = MethodicalMachine()

    class Machination(object):
        machine = expectedMachine
    machination = Machination()
    with self.assertRaises(AttributeError) as cm:
        machination.machine
    self.assertIn('MethodicalMachine is an implementation detail', str(cm.exception))
    self.assertIs(Machination.machine, expectedMachine)