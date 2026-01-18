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
def test_generate_structured_model_1(self):
    m_class = LP_inactive_index()
    m_class._generate_model()
    model = m = m_class.model
    community_map_object = cmo = detect_communities(model, random_seed=5)
    correct_partition = {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0}
    correct_components = {"b[0].'B[2].c'", "b[0].'c2[1]'", "b[0].'c1[3]'", 'equality_constraint_list[1]', "b[1].'c2[2]'", 'b[1].x', 'b[0].x', 'b[0].y', 'b[0].z', "b[0].'obj[2]'", "b[1].'c1[2]'"}
    structured_model = cmo.generate_structured_model()
    self.assertIsInstance(structured_model, Block)
    all_components = set([str(component) for component in structured_model.component_data_objects(ctype=(Var, Constraint, Objective, ConstraintList), active=cmo.use_only_active_components, descend_into=True)])
    if cmo.graph_partition == correct_partition:
        self.assertEqual(2, len(cmo.community_map), len(list(structured_model.component_data_objects(ctype=Block, descend_into=True))))
        self.assertEqual(all_components, correct_components)
        for objective in structured_model.component_data_objects(ctype=Objective, descend_into=True):
            objective_expr = str(objective.expr)
        correct_objective_expr = '- b[0].x + b[0].y + b[0].z'
        self.assertEqual(correct_objective_expr, objective_expr)
    self.assertEqual(len(correct_partition), len(cmo.graph_partition))
    self.assertEqual(len(correct_components), len(all_components))