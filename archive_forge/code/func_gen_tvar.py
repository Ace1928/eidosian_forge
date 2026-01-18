from torch.fx.experimental.migrate_gradual_types.constraint import TVar, DVar, BinConstraintD, \
from torch.fx.experimental.migrate_gradual_types.operation import op_leq
def gen_tvar(curr):
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
    curr += 1
    return (TVar(curr), curr)