from torch.fx.experimental.migrate_gradual_types.constraint import TVar, DVar, BinConstraintD, \
from torch.fx.experimental.migrate_gradual_types.operation import op_leq
def gen_dvar(curr):
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
    curr += 1
    return (DVar(curr), curr)