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
def diagnose_imprecision(offender):
    list_msg = '\n\nFor Numba to be able to compile a list, the list must have a known and\nprecise type that can be inferred from the other variables. Whilst sometimes\nthe type of empty lists can be inferred, this is not always the case, see this\ndocumentation for help:\n\nhttps://numba.readthedocs.io/en/stable/user/troubleshoot.html#my-code-has-an-untyped-list-problem\n'
    if offender is not None:
        if hasattr(offender, 'value'):
            if hasattr(offender.value, 'op'):
                if offender.value.op == 'build_list':
                    return list_msg
                elif offender.value.op == 'call':
                    try:
                        call_name = offender.value.func.name
                        offender = find_offender(call_name)
                        if isinstance(offender.value, ir.Global):
                            if offender.value.name == 'list':
                                return list_msg
                    except (AttributeError, KeyError):
                        pass
    return ''