from numba.core import types, cgutils
from numba.core.imputils import (
@lower_builtin(zip, types.VarArg(types.Any))
def make_zip_object(context, builder, sig, args):
    zip_type = sig.return_type
    assert len(args) == len(zip_type.source_types)
    zipobj = context.make_helper(builder, zip_type)
    for i, (arg, srcty) in enumerate(zip(args, sig.args)):
        zipobj[i] = call_getiter(context, builder, srcty, arg)
    res = zipobj._getvalue()
    return impl_ret_new_ref(context, builder, sig.return_type, res)