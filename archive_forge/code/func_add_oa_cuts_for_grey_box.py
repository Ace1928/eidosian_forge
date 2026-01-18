from math import copysign
from pyomo.core import minimize, value
import pyomo.core.expr as EXPR
from pyomo.contrib.gdpopt.util import time_code
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, MCPP_Error
def add_oa_cuts_for_grey_box(target_model, jacobians_model, config, objective_sense, mip_iter, cb_opt=None):
    sign_adjust = -1 if objective_sense == minimize else 1
    if config.add_slack:
        slack_var = target_model.MindtPy_utils.cuts.slack_vars.add()
    for target_model_grey_box, jacobian_model_grey_box in zip(target_model.MindtPy_utils.grey_box_list, jacobians_model.MindtPy_utils.grey_box_list):
        jacobian_matrix = jacobian_model_grey_box.get_external_model().evaluate_jacobian_outputs().toarray()
        for index, output in enumerate(target_model_grey_box.outputs.values()):
            dual_value = jacobians_model.dual[jacobian_model_grey_box][output.name.replace('outputs', 'output_constraints')]
            target_model.MindtPy_utils.cuts.oa_cuts.add(expr=copysign(1, sign_adjust * dual_value) * (sum((jacobian_matrix[index][var_index] * (var - value(var)) for var_index, var in enumerate(target_model_grey_box.inputs.values()))) - (output - value(output))) - (slack_var if config.add_slack else 0) <= 0)