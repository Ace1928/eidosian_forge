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
def get_fused_cname(fused_cname, orig_cname):
    """
    Given the fused cname id and an original cname, return a specialized cname
    """
    assert fused_cname and orig_cname
    return StringEncoding.EncodedString('%s%s%s' % (Naming.fused_func_prefix, fused_cname, orig_cname))