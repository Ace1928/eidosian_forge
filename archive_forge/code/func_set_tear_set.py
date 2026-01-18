from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def set_tear_set(self, tset):
    """
        Set a custom tear set to be used when running the decomposition

        The procedure will use this custom tear set instead of finding
        its own, thus it can save some time. Additionally, this will be
        useful for knowing which edges will need guesses.

        Arguments
        ---------
            tset
                A list of Arcs representing edges to tear

        While this method makes things more convenient, all it does is:

            `self.options["tear_set"] = tset`
        """
    self.options['tear_set'] = tset