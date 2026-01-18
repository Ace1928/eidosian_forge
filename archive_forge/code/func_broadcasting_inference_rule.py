import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
@register_inference_rule(operator.mul)
@register_inference_rule(torch.ne)
@register_inference_rule('ne')
@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def broadcasting_inference_rule(n: Node, symbols, constraints, counter):
    op_code = None
    if n.target == operator.add or n.target == torch.add:
        op_code = op_add
    elif n.target == operator.mul:
        op_code = op_mul
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        if isinstance(symbols[n.args[0]], TVar) and isinstance(symbols[n.args[1]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            e2 = symbols[n.args[1]]
            return gen_broadcasting_constraints(e1, e2, symbols, counter, my_output)
        else:
            raise NotImplementedError('Method not yet implemented')
    elif isinstance(n.args[0], Node) and isinstance(n.args[1], (int, float)):
        if isinstance(symbols[n.args[0]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            return ([BinConstraintT(my_output, e1, op_eq)], counter)
        elif isinstance(symbols[n.args[0]], DVar):
            my_output, counter = gen_dvar(counter)
            symbols[n] = my_output
            e1 = symbols[n.args[0]]
            c = Conj([BinConstraintD(my_output, BinConstraintD(e1, n.args[1], op_code), op_eq), BinConstraintD(0, my_output, op_leq)])
            return ([c], counter)
    elif isinstance(n.args[1], Node) and isinstance(n.args[0], (int, float)):
        if isinstance(symbols[n.args[1]], TVar):
            my_output, counter = gen_tvar(counter)
            symbols[n] = my_output
            e2 = symbols[n.args[1]]
            return ([BinConstraintT(my_output, e2, op_eq)], counter)
        elif isinstance(symbols[n.args[1]], DVar):
            my_output, counter = gen_dvar(counter)
            symbols[n] = my_output
            e2 = symbols[n.args[1]]
            c = Conj([BinConstraintD(my_output, BinConstraintD(e2, n.args[0], op_code), op_eq), BinConstraintD(0, my_output, op_leq)])
            return ([c], counter)
        else:
            raise NotImplementedError('Method not yet implemented')
    else:
        raise NotImplementedError('Addition not yet implemented')