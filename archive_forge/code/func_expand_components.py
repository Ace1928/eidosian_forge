import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core import Suffix, Var, Constraint, Piecewise, Block
from pyomo.core import Expression, Param
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.block import IndexedBlock, SortComponents
from pyomo.dae import ContinuousSet, DAE_Error
from pyomo.common.formatting import tostr
from io import StringIO
def expand_components(block):
    """
    Loop over block components and try expanding them. If expansion fails
    then save the component and try again later. This function has some
    built-in robustness for block-hierarchical models with circular
    references but will not work for all cases.
    """
    expansion_map = ComponentMap()
    redo_expansion = list()
    for blk in block.component_objects(Block, descend_into=True):
        missing_idx = set(blk.index_set()) - set(blk._data.keys())
        if missing_idx:
            blk._dae_missing_idx = missing_idx
    try:
        buf = StringIO()
        with LoggingIntercept(buf, 'pyomo.core', logging.ERROR):
            for c in block.component_objects(descend_into=True, sort=SortComponents.declOrder):
                try:
                    update_contset_indexed_component(c, expansion_map)
                except AttributeError:
                    redo_expansion.append(c)
            N = len(redo_expansion)
            while N:
                for i in range(N):
                    c = redo_expansion.pop()
                    try:
                        expansion_map[c](c)
                    except AttributeError:
                        redo_expansion.append(c)
                if len(redo_expansion) == N:
                    raise DAE_Error('Unable to fully discretize %s. Possible circular references detected between components %s. Reformulate your model to remove circular references or apply a discretization transformation before linking blocks together.' % (block, tostr(redo_expansion)))
                N = len(redo_expansion)
    except Exception:
        logger.error(buf.getvalue())
        raise