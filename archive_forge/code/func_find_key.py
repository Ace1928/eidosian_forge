import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
def find_key(self, keys: typing.Iterable[str]) -> typing.Optional[str]:
    """
        Find the first mapping key in a sequence of names (keys).

        Args:
          keys (iterable): The keys are the R classes (the last being the
            most distant ancestor class)
        Returns:
           None if no mapping key.
        """
    for k in keys:
        if k in self._map:
            return k
    return None