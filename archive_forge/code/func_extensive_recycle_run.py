import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def extensive_recycle_run(self, tear_method, tol_type):
    rel = tol_type == 'rel'
    m = self.extensive_recycle_model()

    def function(unit):
        unit.initialize()
    seq = SequentialDecomposition(tear_method=tear_method, tol_type=tol_type)
    tset = [m.stream_splitter_to_mixer]
    seq.set_tear_set(tset)
    splitter_to_mixer_guess = {'flow': {'A': [(m.stream_splitter_to_mixer, 0)], 'B': [(m.stream_splitter_to_mixer, 0)], 'C': [(m.stream_splitter_to_mixer, 0)]}, 'mass': [(m.stream_splitter_to_mixer, 0)], 'expr_idx': {'A': [(m.stream_splitter_to_mixer, 0)], 'B': [(m.stream_splitter_to_mixer, 0)], 'C': [(m.stream_splitter_to_mixer, 0)]}, 'expr': [(m.stream_splitter_to_mixer, 0)], 'temperature': 450, 'pressure': 128}
    seq.set_guesses_for(m.mixer.inlet, splitter_to_mixer_guess)
    seq.run(m, function)
    self.check_recycle_model(m, rel=rel)
    if rel:
        s = value(m.prod.inlet.mass)
        d = value(m.feed.outlet.mass)
        self.assertAlmostEqual((s - d) / s, 0, places=5)
    else:
        self.assertAlmostEqual(value(m.prod.inlet.mass), value(m.feed.outlet.mass), places=5)