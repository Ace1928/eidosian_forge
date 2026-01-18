import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,
@lower_cast(types.Optional, types.Optional)
def optional_to_optional(context, builder, fromty, toty, val):
    """
    The handling of optional->optional cast must be special cased for
    correct propagation of None value.  Given type T and U. casting of
    T? to U? (? denotes optional) should always succeed.   If the from-value
    is None, the None value the casted value (U?) should be None; otherwise,
    the from-value is casted to U. This is different from casting T? to U,
    which requires the from-value must not be None.
    """
    optval = context.make_helper(builder, fromty, value=val)
    validbit = cgutils.as_bool_bit(builder, optval.valid)
    outoptval = context.make_helper(builder, toty)
    with builder.if_else(validbit) as (is_valid, is_not_valid):
        with is_valid:
            outoptval.valid = cgutils.true_bit
            outoptval.data = context.cast(builder, optval.data, fromty.type, toty.type)
        with is_not_valid:
            outoptval.valid = cgutils.false_bit
            outoptval.data = cgutils.get_null_value(outoptval.data.type)
    return outoptval._getvalue()