import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(operator.getitem, types.UniTuple, types.intp)
@lower_builtin(operator.getitem, types.UniTuple, types.uintp)
@lower_builtin(operator.getitem, types.NamedUniTuple, types.intp)
@lower_builtin(operator.getitem, types.NamedUniTuple, types.uintp)
def getitem_unituple(context, builder, sig, args):
    tupty, _ = sig.args
    tup, idx = args
    errmsg_oob = ('tuple index out of range',)
    if len(tupty) == 0:
        with builder.if_then(cgutils.true_bit):
            context.call_conv.return_user_exc(builder, IndexError, errmsg_oob)
        res = context.get_constant_null(sig.return_type)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    else:
        bbelse = builder.append_basic_block('switch.else')
        bbend = builder.append_basic_block('switch.end')
        switch = builder.switch(idx, bbelse)
        with builder.goto_block(bbelse):
            context.call_conv.return_user_exc(builder, IndexError, errmsg_oob)
        lrtty = context.get_value_type(tupty.dtype)
        with builder.goto_block(bbend):
            phinode = builder.phi(lrtty)
        for i in range(tupty.count):
            ki = context.get_constant(types.intp, i)
            bbi = builder.append_basic_block('switch.%d' % i)
            switch.add_case(ki, bbi)
            kin = context.get_constant(types.intp, -tupty.count + i)
            switch.add_case(kin, bbi)
            with builder.goto_block(bbi):
                value = builder.extract_value(tup, i)
                builder.branch(bbend)
                phinode.add_incoming(value, bbi)
        builder.position_at_end(bbend)
        res = phinode
        assert sig.return_type == tupty.dtype
        return impl_ret_borrowed(context, builder, sig.return_type, res)