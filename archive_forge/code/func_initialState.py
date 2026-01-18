from itertools import chain
@initialState.setter
def initialState(self, state):
    """
        Set this automaton's initial state.  Raises a ValueError if
        this automaton already has an initial state.
        """
    if self._initialState is not _NO_STATE:
        raise ValueError('initial state already set to {}'.format(self._initialState))
    self._initialState = state