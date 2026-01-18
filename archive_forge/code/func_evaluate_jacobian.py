from collections import namedtuple
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.config import ConfigBlock
from pyomo.util.subsystems import create_subsystem_block
def evaluate_jacobian(self, x0):
    sparse_jac = super().evaluate_jacobian(x0)
    dense_jac = sparse_jac.toarray()
    return dense_jac