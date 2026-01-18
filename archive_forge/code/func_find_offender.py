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
def find_offender(name, exhaustive=False):
    offender = None
    for block in self.func_ir.blocks.values():
        offender = block.find_variable_assignment(name)
        if offender is not None:
            if not exhaustive:
                break
            try:
                hasattr(offender.value, 'name')
                offender_value = offender.value.name
            except (AttributeError, KeyError):
                break
            orig_offender = offender
            if offender_value.startswith('$'):
                offender = find_offender(offender_value, exhaustive=exhaustive)
                if offender is None:
                    offender = orig_offender
            break
    return offender