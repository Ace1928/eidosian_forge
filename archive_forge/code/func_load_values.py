from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def load_values(self, port, default, fixed, use_guesses):
    sources = port.sources()
    for name, index, obj in port.iter_vars(fixed=False, names=True):
        evars = None
        if port.is_extensive(name):
            evars = [arc.expanded_block.component(name) for arc in sources]
            if evars[0] is None:
                evars = None
            else:
                try:
                    for j in range(len(evars)):
                        evars[j] = evars[j][index]
                except AttributeError:
                    pass
        if evars is not None:
            for evar in evars:
                if evar.is_fixed():
                    continue
                self.check_value_fix(port, evar, default, fixed, use_guesses, extensive=True)
            self.combine_and_fix(port, name, obj, evars, fixed)
        elif obj.is_expression_type():
            for var in identify_variables(obj, include_fixed=False):
                self.check_value_fix(port, var, default, fixed, use_guesses)
        else:
            self.check_value_fix(port, obj, default, fixed, use_guesses)