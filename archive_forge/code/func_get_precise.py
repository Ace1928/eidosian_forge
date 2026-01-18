from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, errors
def get_precise(self):
    """
        Return precise function type if possible.
        """
    for dispatcher in self.dispatchers:
        for cres in dispatcher.overloads.values():
            sig = types.unliteral(cres.signature)
            return FunctionType(sig)
    return self