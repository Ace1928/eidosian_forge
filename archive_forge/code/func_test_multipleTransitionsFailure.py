from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_multipleTransitionsFailure(self):
    """
        A L{MethodicalMachine} can only have one transition per start/event
        pair.
        """

    class WillFail(object):
        m = MethodicalMachine()

        @m.state(initial=True)
        def start(self):
            """We start here."""

        @m.state()
        def end(self):
            """Rainbows end."""

        @m.input()
        def event(self):
            """An event."""
        start.upon(event, enter=end, outputs=[])
        with self.assertRaises(ValueError):
            start.upon(event, enter=end, outputs=[])