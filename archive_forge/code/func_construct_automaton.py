from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine
def construct_automaton(self):
    """
        Construct the automaton based on the set of reduction rules of the system.

        Automata Design:
        The accept states of the automaton are the proper prefixes of the left hand side of the rules.
        The complete left hand side of the rules are the dead states of the automaton.

        """
    self._add_to_automaton(self.rules)