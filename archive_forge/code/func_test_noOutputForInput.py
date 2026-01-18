from .._core import Automaton, NoTransition
from unittest import TestCase
def test_noOutputForInput(self):
    """
        L{Automaton.outputForInput} raises L{NoTransition} if no
        transition for that input is defined.
        """
    a = Automaton()
    self.assertRaises(NoTransition, a.outputForInput, 'no-state', 'no-symbol')