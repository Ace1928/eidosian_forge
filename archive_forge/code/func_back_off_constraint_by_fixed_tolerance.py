from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def back_off_constraint_by_fixed_tolerance(cut, transBlock_rHull, bigm_to_hull_map, opt, stream_solver, TOL):
    """Makes cut more conservative by absolute tolerance TOL

    Parameters
    ----------
    cut: the cut to be made more conservative, a Constraint
    transBlock_rHull: the relaxed hull model's transformation Block. Ignored by
                      this callback
    bigm_to_hull_map: Dictionary mapping ids of bigM variables to the
                      corresponding variables on the relaxed hull instance.
                      Ignored by this callback.
    opt: SolverFactory object. Ignored by this callback
    stream_solver: Whether or not to set tee=True while solving. Ignored by
                   this callback
    TOL: An absolute tolerance to be added to make cut more conservative.
    """
    cut._body += TOL