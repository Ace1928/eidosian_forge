import sys
import string
def add_transition_list(self, list_input_symbols, state, action=None, next_state=None):
    """This adds the same transition for a list of input symbols.
        You can pass a list or a string. Note that it is handy to use
        string.digits, string.whitespace, string.letters, etc. to add
        transitions that match character classes.

        The action may be set to None in which case the process() method will
        ignore the action and only set the next_state. The next_state may be
        set to None in which case the current state will be unchanged. """
    if next_state is None:
        next_state = state
    for input_symbol in list_input_symbols:
        self.add_transition(input_symbol, state, action, next_state)