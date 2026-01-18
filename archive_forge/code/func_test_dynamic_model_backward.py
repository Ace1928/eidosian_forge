import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_dynamic_model_backward(self):
    m = make_dynamic_model(nfe=5, scheme='BACKWARD')
    m.height[0].fix()
    constraints = list(m.component_data_objects(pyo.Constraint, active=True))
    variables = list(_generate_variables_in_constraints(constraints))
    con_coord_map = ComponentMap(((con, i) for i, con in enumerate(constraints)))
    var_coord_map = ComponentMap(((var, i) for i, var in enumerate(variables)))
    coo = get_structural_incidence_matrix(variables, constraints)
    row_blocks, col_blocks = get_independent_submatrices(coo)
    rc_blocks = [(tuple(rows), tuple(cols)) for rows, cols in zip(row_blocks, col_blocks)]
    self.assertEqual(len(rc_blocks), 2)
    t0_var_coords = {var_coord_map[m.flow_out[0]], var_coord_map[m.dhdt[0]], var_coord_map[m.flow_in[0]]}
    t0_con_coords = {con_coord_map[m.flow_out_eqn[0]], con_coord_map[m.diff_eqn[0]]}
    var_blocks = [tuple(sorted(t0_var_coords)), tuple((i for i in range(len(variables)) if i not in t0_var_coords))]
    con_blocks = [tuple(sorted(t0_con_coords)), tuple((i for i in range(len(constraints)) if i not in t0_con_coords))]
    target_blocks = [(tuple(rows), tuple(cols)) for rows, cols in zip(con_blocks, var_blocks)]
    target_blocks = list(sorted(target_blocks))
    rc_blocks = list(sorted(rc_blocks))
    self.assertEqual(target_blocks, rc_blocks)