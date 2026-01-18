import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def check_var(name):
    tv = self.typevars[name]
    if not tv.defined:
        if raise_errors:
            offender = find_offender(name)
            val = getattr(offender, 'value', 'unknown operation')
            loc = getattr(offender, 'loc', ir.unknown_loc)
            msg = "Type of variable '%s' cannot be determined, operation: %s, location: %s"
            raise TypingError(msg % (var, val, loc), loc)
        else:
            typdict[var] = types.unknown
            return
    tp = tv.getone()
    if isinstance(tp, types.UndefinedFunctionType):
        tp = tp.get_precise()
    if not tp.is_precise():
        offender = find_offender(name, exhaustive=True)
        msg = "Cannot infer the type of variable '%s'%s, have imprecise type: %s. %s"
        istmp = ' (temporary variable)' if var.startswith('$') else ''
        loc = getattr(offender, 'loc', ir.unknown_loc)
        extra_msg = diagnose_imprecision(offender)
        if raise_errors:
            raise TypingError(msg % (var, istmp, tp, extra_msg), loc)
        else:
            typdict[var] = types.unknown
            return
    else:
        typdict[var] = tp