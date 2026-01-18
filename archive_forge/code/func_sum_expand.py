import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def sum_expand(self, args, kws):
    """
    sum can be called with or without an axis parameter, and with or without
    a dtype parameter
    """
    pysig = None
    if 'axis' in kws and 'dtype' not in kws:

        def sum_stub(axis):
            pass
        pysig = utils.pysignature(sum_stub)
        args = list(args) + [kws['axis']]
    elif 'dtype' in kws and 'axis' not in kws:

        def sum_stub(dtype):
            pass
        pysig = utils.pysignature(sum_stub)
        args = list(args) + [kws['dtype']]
    elif 'dtype' in kws and 'axis' in kws:

        def sum_stub(axis, dtype):
            pass
        pysig = utils.pysignature(sum_stub)
        args = list(args) + [kws['axis'], kws['dtype']]
    args_len = len(args)
    assert args_len <= 2
    if args_len == 0:
        out = signature(_expand_integer(self.this.dtype), *args, recvr=self.this)
    elif args_len == 1 and 'dtype' not in kws:
        if self.this.ndim == 1:
            return_type = self.this.dtype
        else:
            return_type = types.Array(dtype=_expand_integer(self.this.dtype), ndim=self.this.ndim - 1, layout='C')
        out = signature(return_type, *args, recvr=self.this)
    elif args_len == 1 and 'dtype' in kws:
        from .npydecl import parse_dtype
        dtype, = args
        dtype = parse_dtype(dtype)
        out = signature(dtype, *args, recvr=self.this)
    elif args_len == 2:
        from .npydecl import parse_dtype
        dtype = parse_dtype(args[1])
        return_type = dtype
        if self.this.ndim != 1:
            return_type = types.Array(dtype=return_type, ndim=self.this.ndim - 1, layout='C')
        out = signature(return_type, *args, recvr=self.this)
    else:
        pass
    return out.replace(pysig=pysig)