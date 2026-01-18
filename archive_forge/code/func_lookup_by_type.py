import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
def lookup_by_type(self, typ):
    """Look up the registered formatter for a type.

        Parameters
        ----------
        typ : type or '__module__.__name__' string for a type

        Returns
        -------
        f : callable
            The registered formatting callable for the type.

        Raises
        ------
        KeyError if the type has not been registered.
        """
    if isinstance(typ, str):
        typ_key = tuple(typ.rsplit('.', 1))
        if typ_key not in self.deferred_printers:
            for cls in self.type_printers:
                if _mod_name_key(cls) == typ_key:
                    return self.type_printers[cls]
        else:
            return self.deferred_printers[typ_key]
    else:
        for cls in pretty._get_mro(typ):
            if cls in self.type_printers or self._in_deferred_types(cls):
                return self.type_printers[cls]
    raise KeyError('No registered printer for {0!r}'.format(typ))