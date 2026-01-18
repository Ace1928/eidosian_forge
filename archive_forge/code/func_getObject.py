from weakref import ref as weakref_ref
def getObject(self, symbol):
    """
        Return the object corresponding to a symbol
        """
    if symbol in self.bySymbol:
        return self.bySymbol[symbol]
    elif symbol in self.aliases:
        return self.aliases[symbol]
    else:
        return SymbolMap.UnknownSymbol