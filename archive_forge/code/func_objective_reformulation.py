from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_oa_cuts_for_grey_box
def objective_reformulation(self):
    MindtPy = self.working_model.MindtPy_utils
    config = self.config
    self.process_objective(update_var_con_list=config.add_regularization is None)
    if MindtPy.objective_list[0].expr.polynomial_degree() in self.mip_objective_polynomial_degree and config.add_regularization is not None:
        MindtPy.objective_list[0].activate()
        MindtPy.objective_constr.deactivate()
        MindtPy.objective.deactivate()