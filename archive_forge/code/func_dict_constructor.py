from numba.core import types
from numba.core.imputils import lower_builtin
@lower_builtin(dict, types.IterableType)
def dict_constructor(context, builder, sig, args):
    from numba.typed import Dict
    dicttype = sig.return_type
    kt, vt = (dicttype.key_type, dicttype.value_type)

    def dict_impl(iterable):
        res = Dict.empty(kt, vt)
        for k, v in iterable:
            res[k] = v
        return res
    return context.compile_internal(builder, dict_impl, sig, args)