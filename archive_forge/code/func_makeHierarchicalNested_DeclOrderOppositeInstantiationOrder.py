from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeHierarchicalNested_DeclOrderOppositeInstantiationOrder():
    """Here, we declare the Blocks in the opposite order. This means that
    decl order will be *opposite* instantiation order, which means that we
    can break our targets preprocessing without even using targets if we
    are not correctly identifying what is nested in what!"""
    m = ConcreteModel()
    m.I = RangeSet(1, 4)
    m.x = Var(m.I, bounds=(-2, 6))
    m.disjunction_block = Block()
    m.disjunct_block = Block()
    instantiate_hierarchical_nested_model(m)
    return m