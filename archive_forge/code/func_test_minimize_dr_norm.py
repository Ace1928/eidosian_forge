import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
@unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
def test_minimize_dr_norm(self):
    m = ConcreteModel()
    m.p1 = Param(initialize=0, mutable=True)
    m.p2 = Param(initialize=0, mutable=True)
    m.z1 = Var(initialize=0, bounds=(0, 1))
    m.z2 = Var(initialize=0, bounds=(0, 1))
    m.working_model = ConcreteModel()
    m.working_model.util = Block()
    m.working_model.util.second_stage_variables = [m.z1, m.z2]
    m.working_model.util.uncertain_params = [m.p1, m.p2]
    m.working_model.util.first_stage_variables = []
    m.working_model.util.state_vars = []
    m.working_model.util.first_stage_variables = []
    config = Bunch()
    config.decision_rule_order = 1
    config.objective_focus = ObjectiveType.nominal
    config.global_solver = SolverFactory('baron')
    config.uncertain_params = m.working_model.util.uncertain_params
    config.tee = False
    config.solve_master_globally = True
    config.time_limit = None
    config.progress_logger = logging.getLogger(__name__)
    add_decision_rule_variables(model_data=m, config=config)
    add_decision_rule_constraints(model_data=m, config=config)
    master = ConcreteModel()
    master.scenarios = Block(NonNegativeIntegers, NonNegativeIntegers)
    master.scenarios[0, 0].transfer_attributes_from(m.working_model.clone())
    master.scenarios[0, 0].first_stage_objective = 0
    master.scenarios[0, 0].second_stage_objective = Expression(expr=(master.scenarios[0, 0].util.second_stage_variables[0] - 1) ** 2 + (master.scenarios[0, 0].util.second_stage_variables[1] - 1) ** 2)
    master.obj = Objective(expr=master.scenarios[0, 0].second_stage_objective)
    master_data = MasterProblemData()
    master_data.master_model = master
    master_data.master_model.const_efficiency_applied = False
    master_data.master_model.linear_efficiency_applied = False
    master_data.iteration = 0
    master_data.timing = TimingData()
    with time_code(master_data.timing, 'main', is_main_timer=True):
        results, success = minimize_dr_vars(model_data=master_data, config=config)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal, msg='Minimize dr norm did not solve to optimality.')
        self.assertTrue(success, msg=f'DR polishing success {success}, expected True.')