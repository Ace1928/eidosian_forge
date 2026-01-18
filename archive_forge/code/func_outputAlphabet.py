from itertools import chain
def outputAlphabet(self):
    """
        The full set of symbols which can be produced by this automaton.
        """
    return set(chain.from_iterable((outputSymbols for inState, inputSymbol, outState, outputSymbols in self._transitions)))