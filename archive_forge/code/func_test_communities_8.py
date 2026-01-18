import logging
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import networkx_available, matplotlib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.contrib.community_detection.detection import (
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple
def test_communities_8(self):
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.community_detection', logging.ERROR):
        detect_communities(ConcreteModel())
    self.assertIn('in detect_communities: Empty community map was returned', output.getvalue())
    with LoggingIntercept(output, 'pyomo.contrib.community_detection', logging.WARNING):
        detect_communities(one_community_model())
    self.assertIn('Community detection found that with the given parameters, the model could not be decomposed - only one community was found', output.getvalue())
    model = 'foo'
    with self.assertRaisesRegex(TypeError, "Invalid model: 'model=%s' - model must be an instance of ConcreteModel" % model):
        detect_communities(model)
    model = create_model_6()
    type_of_community_map = 'foo'
    with self.assertRaisesRegex(TypeError, "Invalid value for type_of_community_map: 'type_of_community_map=%s' - Valid values: 'bipartite', 'constraint', 'variable'" % type_of_community_map):
        detect_communities(model, type_of_community_map=type_of_community_map)
    with_objective = 'foo'
    with self.assertRaisesRegex(TypeError, "Invalid value for with_objective: 'with_objective=%s' - with_objective must be a Boolean" % with_objective):
        detect_communities(model, with_objective=with_objective)
    weighted_graph = 'foo'
    with self.assertRaisesRegex(TypeError, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph must be a Boolean" % weighted_graph):
        detect_communities(model, weighted_graph=weighted_graph)
    random_seed = 'foo'
    with self.assertRaisesRegex(TypeError, "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed):
        detect_communities(model, random_seed=random_seed)
    random_seed = -1
    with self.assertRaisesRegex(ValueError, "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed):
        detect_communities(model, random_seed=random_seed)
    use_only_active_components = 'foo'
    with self.assertRaisesRegex(TypeError, "Invalid value for use_only_active_components: 'use_only_active_components=%s' - use_only_active_components must be True or None" % use_only_active_components):
        detect_communities(model, use_only_active_components=use_only_active_components)