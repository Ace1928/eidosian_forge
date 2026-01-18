from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def return_optional_value(self, builder, retty, valty, value):
    if valty == types.none:
        self.return_native_none(builder)
    elif retty == valty:
        optval = self.context.make_helper(builder, retty, value=value)
        validbit = cgutils.as_bool_bit(builder, optval.valid)
        with builder.if_then(validbit):
            retval = self.context.get_return_value(builder, retty.type, optval.data)
            self.return_value(builder, retval)
        self.return_native_none(builder)
    elif not isinstance(valty, types.Optional):
        if valty != retty.type:
            value = self.context.cast(builder, value, fromty=valty, toty=retty.type)
        retval = self.context.get_return_value(builder, retty.type, value)
        self.return_value(builder, retval)
    else:
        raise NotImplementedError('returning {0} for {1}'.format(valty, retty))