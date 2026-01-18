import logging
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def print_expr_canonical(obj, x, output, object_symbol_dictionary, variable_symbol_dictionary, is_objective, column_order, file_determinism, force_objective_constant=False):
    try:
        return self._print_expr_canonical(x=x, output=output, object_symbol_dictionary=object_symbol_dictionary, variable_symbol_dictionary=variable_symbol_dictionary, is_objective=is_objective, column_order=column_order, file_determinism=file_determinism, force_objective_constant=force_objective_constant)
    except KeyError as e:
        _id = e.args[0]
        _var = None
        if x.linear_vars:
            for v in x.linear_vars:
                if id(v) == _id:
                    _var = v
                    break
        if _var is None and x.quadratic_vars:
            for v in x.quadratic_vars:
                v = [_v for _v in v if id(_v) == _id]
                if v:
                    _var = v[0]
                    break
        if _var is not None:
            logger.error('Model contains an expression (%s) that contains a variable (%s) that is not attached to an active block on the submodel being written' % (obj.name, _var.name))
        raise