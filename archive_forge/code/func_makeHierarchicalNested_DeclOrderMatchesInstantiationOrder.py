from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeHierarchicalNested_DeclOrderMatchesInstantiationOrder():
    """Here, we put the disjunctive components on Blocks, but we do it in the
    same order that we declared the blocks, that is, on each block, decl order
    matches instantiation order."""
    m = ConcreteModel()
    m.I = RangeSet(1, 4)
    m.x = Var(m.I, bounds=(-2, 6))
    m.disjunct_block = Block()
    m.disjunction_block = Block()
    instantiate_hierarchical_nested_model(m)
    return m