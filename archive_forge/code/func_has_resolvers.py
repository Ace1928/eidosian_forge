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
@property
def has_resolvers(self) -> bool:
    """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
        """
    return bool(len(self.resolvers))