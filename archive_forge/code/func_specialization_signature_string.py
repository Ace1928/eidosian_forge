from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
def specialization_signature_string(fused_compound_type, fused_to_specific):
    """
    Return the signature for a specialization of a fused type. e.g.

        floating[:] ->
            'float' or 'double'

        cdef fused ft:
            float[:]
            double[:]

        ft ->
            'float[:]' or 'double[:]'

        integral func(floating) ->
            'int (*func)(float)' or ...
    """
    fused_types = fused_compound_type.get_fused_types()
    if len(fused_types) == 1:
        fused_type = fused_types[0]
    else:
        fused_type = fused_compound_type
    return fused_type.specialize(fused_to_specific).typeof_name()