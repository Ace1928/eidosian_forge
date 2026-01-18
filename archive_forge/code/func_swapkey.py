from __future__ import annotations
from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import TypeVar
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.errors import UndefinedVariableError
def swapkey(self, old_key: str, new_key: str, new_value=None) -> None:
    """
        Replace a variable name, with a potentially new value.

        Parameters
        ----------
        old_key : str
            Current variable name to replace
        new_key : str
            New variable name to replace `old_key` with
        new_value : object
            Value to be replaced along with the possible renaming
        """
    if self.has_resolvers:
        maps = self.resolvers.maps + self.scope.maps
    else:
        maps = self.scope.maps
    maps.append(self.temps)
    for mapping in maps:
        if old_key in mapping:
            mapping[new_key] = new_value
            return