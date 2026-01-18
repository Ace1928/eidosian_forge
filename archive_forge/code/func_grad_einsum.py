from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_einsum(argnum, ans, operands_, kwargs):
    result_meta = anp.metadata(operands_[argnum])

    def vjp(g):
        operands = operands_
        if isinstance(operands[0], string_types):
            in_subs, out_subs, _ = anp.parse_einsum_input(*operands)
            string, operands = (operands[0], operands[1:])
            in_subs_list = in_subs.split(',')
            op_num = argnum - 1
            subs_wrt = in_subs_list[op_num]
            rest_of_ops = operands[:op_num] + operands[op_num + 1:]
            rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num + 1:]
            other_named_subs = set(''.join([out_subs] + rest_of_subs))
            naked_summed = [(i, sub) for i, sub in enumerate(subs_wrt) if sub not in other_named_subs]
            if naked_summed:
                naked_summed_dims, ones_subs = zip(*naked_summed)
                ones_subs = ''.join(ones_subs)
                ones = onp.ones(onp.array(operands[op_num].shape)[list(naked_summed_dims)])
                new_input_subs = ','.join([out_subs, ones_subs] + rest_of_subs)
                new_operands = (g, ones) + rest_of_ops
            else:
                new_input_subs = ','.join([out_subs] + rest_of_subs)
                new_operands = (g,) + rest_of_ops
            new_subscripts = new_input_subs + '->' + subs_wrt
            return unbroadcast(anp.einsum(new_subscripts, *new_operands), result_meta)
        else:
            if len(operands) % 2 == 0:
                raise NotImplementedError('Need sublistout argument')
            operands = list(operands)
            rest_of_ops = [operands[-1]] + operands[:argnum] + operands[argnum + 2:-1] + [operands[argnum + 1]]
            return unbroadcast_einsum(anp.einsum(g, *rest_of_ops), result_meta, operands[argnum + 1])
    return vjp