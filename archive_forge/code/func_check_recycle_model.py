import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def check_recycle_model(self, m, rel=False):
    for arc in m.component_data_objects(Arc):
        self.assertTrue(self.is_converged(arc, rel=rel))
    for port in m.component_data_objects(Port):
        self.assertTrue(self.intensive_equal(port, temperature=value(m.feed.outlet.temperature), pressure=value(m.feed.outlet.pressure)))
    if rel:
        for i in m.feed.outlet.flow:
            s = value(m.prod.inlet.flow[i])
            d = value(m.feed.outlet.flow[i])
            self.assertAlmostEqual((s - d) / s, 0, places=5)
        for i in m.feed.outlet.expr_idx:
            s = value(m.prod.inlet.expr_idx[i])
            d = value(m.feed.outlet.expr_idx[i])
            self.assertAlmostEqual((s - d) / s, 0, places=5)
        s = value(m.prod.inlet.expr)
        d = value(m.feed.outlet.expr)
        self.assertAlmostEqual((s - d) / s, 0, places=5)
        for i in m.feed.outlet.expr_idx:
            s = value(-m.prod.actual_var_idx_in[i])
            d = value(m.feed.expr_var_idx_out[i])
            self.assertAlmostEqual((s - d) / s, 0, places=5)
        s = value(-m.prod.actual_var_in)
        d = value(m.feed.expr_var_out)
        self.assertAlmostEqual((s - d) / s, 0, places=5)
    else:
        for i in m.feed.outlet.flow:
            self.assertAlmostEqual(value(m.prod.inlet.flow[i]), value(m.feed.outlet.flow[i]), places=5)
        for i in m.feed.outlet.expr_idx:
            self.assertAlmostEqual(value(m.prod.inlet.expr_idx[i]), value(m.feed.outlet.expr_idx[i]), places=5)
        self.assertAlmostEqual(value(m.prod.inlet.expr), value(m.feed.outlet.expr), places=5)
        for i in m.feed.outlet.expr_idx:
            self.assertAlmostEqual(value(-m.prod.actual_var_idx_in[i]), value(m.feed.expr_var_idx_out[i]), places=5)
        self.assertAlmostEqual(value(-m.prod.actual_var_in), value(m.feed.expr_var_out), places=5)