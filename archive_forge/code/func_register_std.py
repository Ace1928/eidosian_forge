import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
def register_std(self):
    """ Registers all the standard checkers in the given checker.

        If not present already, the standard checkers schema (STD_NAMESPACE) is
        added to the checker's namespace with an empty prefix.
        """
    self._namespace.register(STD_NAMESPACE, '')
    for cond in _ALL_CHECKERS:
        self.register(cond, STD_NAMESPACE, _ALL_CHECKERS[cond])