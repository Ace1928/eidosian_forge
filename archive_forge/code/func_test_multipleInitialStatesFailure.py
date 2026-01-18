from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
def test_multipleInitialStatesFailure(self):
    """
        A L{MethodicalMachine} can only have one initial state.
        """

    class WillFail(object):
        m = MethodicalMachine()

        @m.state(initial=True)
        def firstInitialState(self):
            """The first initial state -- this is OK."""
        with self.assertRaises(ValueError):

            @m.state(initial=True)
            def secondInitialState(self):
                """The second initial state -- results in a ValueError."""