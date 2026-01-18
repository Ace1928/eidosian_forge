from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
@classmethod
def fromBlob(cls, blob):
    self = cls()
    self._restore(blob)
    return self