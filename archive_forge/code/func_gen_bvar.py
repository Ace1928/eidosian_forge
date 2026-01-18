from torch.fx.experimental.migrate_gradual_types.constraint import TVar, DVar, BinConstraintD, \
from torch.fx.experimental.migrate_gradual_types.operation import op_leq
def gen_bvar(curr):
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
    curr += 1
    return (BVar(curr), curr)