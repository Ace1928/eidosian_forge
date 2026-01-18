import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.reshape')
def resolve_reshape(self, ary, args, kws):

    def sentry_shape_scalar(ty):
        if ty in types.number_domain:
            if not isinstance(ty, types.Integer):
                raise TypeError('reshape() arg cannot be {0}'.format(ty))
            return True
        else:
            return False
    assert not kws
    if ary.layout not in 'CF':
        raise TypeError('reshape() supports contiguous array only')
    if len(args) == 1:
        shape, = args
        if sentry_shape_scalar(shape):
            ndim = 1
        else:
            shape = normalize_shape(shape)
            if shape is None:
                return
            ndim = shape.count
        retty = ary.copy(ndim=ndim)
        return signature(retty, shape)
    elif len(args) == 0:
        raise TypeError('reshape() take at least one arg')
    else:
        if any((not sentry_shape_scalar(a) for a in args)):
            raise TypeError('reshape({0}) is not supported'.format(', '.join(map(str, args))))
        retty = ary.copy(ndim=len(args))
        return signature(retty, *args)