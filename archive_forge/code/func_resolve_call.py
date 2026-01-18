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
def resolve_call(self, fnty, pos_args, kw_args):
    """
        Resolve a call to a given function type.  A signature is returned.
        """
    if isinstance(fnty, types.FunctionType):
        return fnty.get_call_type(self, pos_args, kw_args)
    if isinstance(fnty, types.RecursiveCall) and (not self._skip_recursion):
        disp = fnty.dispatcher_type.dispatcher
        pysig, args = disp.fold_argument_types(pos_args, kw_args)
        frame = self.context.callstack.match(disp.py_func, args)
        if frame is None:
            sig = self.context.resolve_function_type(fnty.dispatcher_type, pos_args, kw_args)
            fndesc = disp.overloads[args].fndesc
            qual = qualifying_prefix(fndesc.modname, fndesc.qualname)
            fnty.add_overloads(args, qual, fndesc.uid)
            return sig
        fnid = frame.func_id
        qual = qualifying_prefix(fnid.modname, fnid.func_qualname)
        fnty.add_overloads(args, qual, fnid.unique_id)
        return_type = frame.typeinfer.return_types_from_partial()
        if return_type is None:
            raise TypingError('cannot type infer runaway recursion')
        sig = typing.signature(return_type, *args)
        sig = sig.replace(pysig=pysig)
        frame.add_return_type(return_type)
        return sig
    else:
        return self.context.resolve_function_type(fnty, pos_args, kw_args)