from weakref import ref as weakref_ref
def createSymbol(self, obj, labeler=None, *args):
    """
        Create a symbol for an object with a given labeler.  No
        error checking is done to ensure that the generated symbol
        name is unique.
        """
    if labeler is None:
        if self.default_labeler is not None:
            labeler = self.default_labeler
        else:
            labeler = str
    symbol = labeler(obj, *args)
    self.addSymbol(obj, symbol)
    return symbol