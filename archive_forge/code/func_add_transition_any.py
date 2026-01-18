import sys
import string
def add_transition_any(self, state, action=None, next_state=None):
    """This adds a transition that associates:

                (current_state) --> (action, next_state)

        That is, any input symbol will match the current state.
        The process() method checks the "any" state associations after it first
        checks for an exact match of (input_symbol, current_state).

        The action may be set to None in which case the process() method will
        ignore the action and only set the next_state. The next_state may be
        set to None in which case the current state will be unchanged. """
    if next_state is None:
        next_state = state
    self.state_transitions_any[state] = (action, next_state)