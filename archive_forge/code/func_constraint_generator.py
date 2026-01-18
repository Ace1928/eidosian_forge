import logging
from io import StringIO
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def constraint_generator():
    for block in all_blocks:
        gen_con_repn = getattr(block, '_gen_con_repn', True)
        if not hasattr(block, '_repn'):
            block._repn = ComponentMap()
        block_repn = block._repn
        for constraint_data in block.component_data_objects(Constraint, active=True, sort=sortOrder, descend_into=False):
            if not constraint_data.has_lb() and (not constraint_data.has_ub()):
                assert not constraint_data.equality
                continue
            if constraint_data._linear_canonical_form:
                repn = constraint_data.canonical_form()
            elif gen_con_repn:
                repn = generate_standard_repn(constraint_data.body)
                block_repn[constraint_data] = repn
            else:
                repn = block_repn[constraint_data]
            yield (constraint_data, repn)