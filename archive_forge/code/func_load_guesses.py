from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def load_guesses(self, guesses, port, fixed):
    srcs = port.sources()
    for name, mem in port.vars.items():
        try:
            entry = guesses[port][name]
        except KeyError:
            continue
        if isinstance(entry, dict):
            itr = [(mem[k], entry[k], k) for k in entry]
        elif mem.is_indexed():
            raise TypeError("Guess for indexed member '%s' in port '%s' must map to a dict of indexes" % (name, port.name))
        else:
            itr = [(mem, entry, None)]
        for var, entry, idx in itr:
            if var.is_fixed():
                continue
            has_evars = False
            if port.is_extensive(name):
                for arc, val in entry:
                    if arc not in srcs:
                        raise ValueError("Found a guess for extensive member '%s' on port '%s' using arc '%s' that is not a source of this port" % (name, port.name, arc.name))
                    evar = arc.expanded_block.component(name)
                    if evar is None:
                        break
                    has_evars = True
                    evar = evar[idx]
                    if evar.is_fixed():
                        continue
                    fixed.add(evar)
                    evar.fix(float(val))
            if not has_evars:
                if var.is_expression_type():
                    raise ValueError("Cannot provide guess for expression type member '%s%s' of port '%s', must set current value of variables within expression" % (name, '[%s]' % str(idx) if mem.is_indexed() else '', port.name))
                else:
                    fixed.add(var)
                    var.fix(float(entry))