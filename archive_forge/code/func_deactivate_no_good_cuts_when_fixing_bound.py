from pyomo.contrib.mindtpy.util import calc_jacobians
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_oa_cuts, add_oa_cuts_for_grey_box
def deactivate_no_good_cuts_when_fixing_bound(self, no_good_cuts):
    if self.config.add_no_good_cuts:
        no_good_cuts[len(no_good_cuts)].deactivate()
    if self.config.use_tabu_list:
        self.integer_list = self.integer_list[:-1]