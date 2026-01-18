import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def simple_recycle_run(self, tear_method, tol_type):
    rel = tol_type == 'rel'
    m = self.simple_recycle_model()

    def function(unit):
        unit.initialize()
    seq = SequentialDecomposition(tear_method=tear_method, tol_type=tol_type)
    tset = [m.stream_splitter_to_mixer]
    seq.set_tear_set(tset)
    splitter_to_mixer_guess = {'flow': {'A': 0, 'B': 0, 'C': 0}, 'temperature': 450, 'pressure': 128}
    seq.set_guesses_for(m.mixer.inlet_side_2, splitter_to_mixer_guess)
    m.mixer.expr_var_idx_in_side_2['A'] = 0
    m.mixer.expr_var_idx_in_side_2['B'] = 0
    m.mixer.expr_var_idx_in_side_2['C'] = 0
    m.mixer.expr_var_in_side_2 = 0
    seq.run(m, function)
    self.check_recycle_model(m, rel=rel)