from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def set_guesses_for(self, port, guesses):
    """
        Set the guesses for the given port

        These guesses will be checked for all free variables that are
        encountered during the first pass run. If a free variable has
        no guess, its current value will be used. If its current value
        is None, the default_guess option will be used. If that is None,
        an error will be raised.

        All port variables that are downstream of a non-tear edge will
        already be fixed. If there is a guess for a fixed variable, it
        will be silently ignored.

        The guesses should be a dict that maps the following:

            Port Member Name -> Value

        Or, for indexed members, multiple dicts that map:

            Port Member Name -> Index -> Value

        For extensive members, "Value" must be a list of tuples of the
        form (arc, value) to guess a value for the expanded variable
        of the specified arc. However, if the arc connecting this port
        is a 1-to-1 arc with its peer, then there will be no expanded
        variable for the single arc, so a regular "Value" should be
        provided.

        This dict cannot be used to pass guesses for variables within
        expression type members. Guesses for those variables must be
        assigned to the variable's current value before calling run.

        While this method makes things more convenient, all it does is:

            `self.options["guesses"][port] = guesses`
        """
    self.options['guesses'][port] = guesses