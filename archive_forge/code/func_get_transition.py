import sys
import string
def get_transition(self, input_symbol, state):
    """This returns (action, next state) given an input_symbol and state.
        This does not modify the FSM state, so calling this method has no side
        effects. Normally you do not call this method directly. It is called by
        process().

        The sequence of steps to check for a defined transition goes from the
        most specific to the least specific.

        1. Check state_transitions[] that match exactly the tuple,
            (input_symbol, state)

        2. Check state_transitions_any[] that match (state)
            In other words, match a specific state and ANY input_symbol.

        3. Check if the default_transition is defined.
            This catches any input_symbol and any state.
            This is a handler for errors, undefined states, or defaults.

        4. No transition was defined. If we get here then raise an exception.
        """
    if (input_symbol, state) in self.state_transitions:
        return self.state_transitions[input_symbol, state]
    elif state in self.state_transitions_any:
        return self.state_transitions_any[state]
    elif self.default_transition is not None:
        return self.default_transition
    else:
        raise ExceptionFSM('Transition is undefined: (%s, %s).' % (str(input_symbol), str(state)))