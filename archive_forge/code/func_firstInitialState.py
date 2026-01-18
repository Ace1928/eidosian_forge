from functools import reduce
from unittest import TestCase
from automat._methodical import ArgSpec, _getArgNames, _getArgSpec, _filterArgs
from .. import MethodicalMachine, NoTransition
from .. import _methodical
@m.state(initial=True)
def firstInitialState(self):
    """The first initial state -- this is OK."""