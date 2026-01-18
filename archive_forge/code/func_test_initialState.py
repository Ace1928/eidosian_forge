from .._core import Automaton, NoTransition
from unittest import TestCase
def test_initialState(self):
    """
        L{Automaton.initialState} is a descriptor that sets the initial
        state if it's not yet set, and raises L{ValueError} if it is.

        """
    a = Automaton()
    a.initialState = 'a state'
    self.assertEqual(a.initialState, 'a state')
    with self.assertRaises(ValueError):
        a.initialState = 'another state'