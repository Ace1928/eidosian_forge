import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
@classmethod
def from_object(cls, obj) -> VT:
    """Create an R vector/array from a Python object, if possible.

        An exception :class:`ValueError` will be raised if not possible."""
    try:
        mv = memoryview(obj)
        res = cls.from_memoryview(mv)
    except (TypeError, ValueError):
        try:
            res = cls.from_iterable(obj)
        except ValueError:
            msg = 'The class methods from_memoryview() and from_iterable() both failed to make a {} from an object of class {}'.format(cls, type(obj))
            raise ValueError(msg)
    return res