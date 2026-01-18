import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def raise_UniversalPicklingError(self):
    """
        Raise L{UniversalPicklingError}.
        """
    raise _UniversalPicklingError