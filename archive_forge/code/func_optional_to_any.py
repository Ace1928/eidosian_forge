import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,
@lower_cast(types.Optional, types.Any)
@lower_cast(types.Optional, types.Boolean)
def optional_to_any(context, builder, fromty, toty, val):
    optval = context.make_helper(builder, fromty, value=val)
    validbit = cgutils.as_bool_bit(builder, optval.valid)
    with builder.if_then(builder.not_(validbit), likely=False):
        msg = 'expected %s, got None' % (fromty.type,)
        context.call_conv.return_user_exc(builder, TypeError, (msg,))
    return context.cast(builder, optval.data, fromty.type, toty)