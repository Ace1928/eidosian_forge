from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_badTransitionForCurrentState(self):
    """
        Calling any input method that lacks a transition for the machine's
        current state raises an informative L{NoTransition}.
        """

    class OnlyOnePath(object):
        m = MethodicalMachine()

        @m.state(initial=True)
        def start(self):
            """Start state."""

        @m.state()
        def end(self):
            """End state."""

        @m.input()
        def advance(self):
            """Move from start to end."""

        @m.input()
        def deadEnd(self):
            """A transition from nowhere to nowhere."""
        start.upon(advance, end, [])
    machine = OnlyOnePath()
    with self.assertRaises(NoTransition) as cm:
        machine.deadEnd()
    self.assertIn('deadEnd', str(cm.exception))
    self.assertIn('start', str(cm.exception))
    machine.advance()
    with self.assertRaises(NoTransition) as cm:
        machine.deadEnd()
    self.assertIn('deadEnd', str(cm.exception))
    self.assertIn('end', str(cm.exception))