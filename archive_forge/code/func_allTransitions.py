from itertools import chain
def allTransitions(self):
    """
        All transitions.
        """
    return frozenset(self._transitions)