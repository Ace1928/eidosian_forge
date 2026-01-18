import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m
import logging
def iteration_callback(nlp, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
    logger = logging.getLogger('pyomo')
    constraint_names = nlp.constraint_names()
    residuals = nlp.evaluate_constraints()
    logger.info('      ...Residuals for iteration {}'.format(iter_count))
    for i, nm in enumerate(constraint_names):
        logger.info('      ...{}: {}'.format(nm, residuals[i]))