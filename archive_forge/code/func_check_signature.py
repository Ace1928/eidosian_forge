from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, errors
def check_signature(self, other_sig):
    """Return True if signatures match (up to being precise).
        """
    sig = self.signature
    return self.nargs == len(other_sig.args) and (sig == other_sig or not sig.is_precise())