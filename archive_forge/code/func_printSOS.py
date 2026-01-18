import logging
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.base import (
from pyomo.repn import generate_standard_repn
def printSOS(self, symbol_map, labeler, variable_symbol_map, soscondata, output):
    """
        Prints the SOS constraint associated with the _SOSConstraintData object
        """
    sos_template_string = self.sos_template_string
    if hasattr(soscondata, 'get_items'):
        sos_items = list(soscondata.get_items())
    else:
        sos_items = list(soscondata.items())
    if len(sos_items) == 0:
        return
    level = soscondata.level
    output.append('%s: S%s::\n' % (symbol_map.getSymbol(soscondata, labeler), level))
    for vardata, weight in sos_items:
        weight = _get_bound(weight)
        if weight < 0:
            raise ValueError('Cannot use negative weight %f for variable %s is special ordered set %s ' % (weight, vardata.name, soscondata.name))
        if vardata.fixed:
            raise RuntimeError("SOSConstraint '%s' includes a fixed variable '%s'. This is currently not supported. Deactivate this constraint in order to proceed." % (soscondata.name, vardata.name))
        self._referenced_variable_ids[id(vardata)] = vardata
        output.append(sos_template_string % (variable_symbol_map.getSymbol(vardata), weight))