from itertools import chain
def outputForInput(self, inState, inputSymbol):
    """
        A 2-tuple of (outState, outputSymbols) for inputSymbol.
        """
    for anInState, anInputSymbol, outState, outputSymbols in self._transitions:
        if (inState, inputSymbol) == (anInState, anInputSymbol):
            return (outState, list(outputSymbols))
    raise NoTransition(state=inState, symbol=inputSymbol)