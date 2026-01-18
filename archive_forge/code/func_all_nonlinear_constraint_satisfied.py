from pyomo.contrib.gdpopt.util import time_code, get_main_elapsed_time
from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_ECP_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_ecp_cuts
from pyomo.opt import TerminationCondition as tc
def all_nonlinear_constraint_satisfied(self):
    config = self.config
    MindtPy = self.mip.MindtPy_utils
    nonlinear_constraints = [c for c in MindtPy.nonlinear_constraint_list]
    for nlc in nonlinear_constraints:
        if nlc.has_lb():
            try:
                lower_slack = nlc.lslack()
            except (ValueError, OverflowError) as e:
                config.logger.error(e, exc_info=True)
                lower_slack = -10 * config.ecp_tolerance
            if lower_slack < -config.ecp_tolerance:
                config.logger.debug('MindtPy-ECP continuing as {} has not met the nonlinear constraints satisfaction.\n'.format(nlc))
                return False
        if nlc.has_ub():
            try:
                upper_slack = nlc.uslack()
            except (ValueError, OverflowError) as e:
                config.logger.error(e, exc_info=True)
                upper_slack = -10 * config.ecp_tolerance
            if upper_slack < -config.ecp_tolerance:
                config.logger.debug('MindtPy-ECP continuing as {} has not met the nonlinear constraints satisfaction.\n'.format(nlc))
                return False
    self.primal_bound = self.dual_bound
    config.logger.info('MindtPy-ECP exiting on nonlinear constraints satisfaction. Primal Bound: {} Dual Bound: {}\n'.format(self.primal_bound, self.dual_bound))
    self.best_solution_found = self.mip.clone()
    self.results.solver.termination_condition = tc.optimal
    return True