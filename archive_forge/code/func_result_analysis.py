from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def result_analysis(self, result=None):
    """Calculate FIM from Jacobian information. This is for grid search (combined models) results

        Parameters
        ----------
        result:
            solver status returned by IPOPT
        """
    self.result = result
    self.doe_result = None
    no_param = len(self.parameter_names)
    fim = np.zeros((no_param, no_param))
    Q_all = []
    for par in self.parameter_names:
        Q_all.append(self.jaco_information[par])
    n = len(self.parameter_names)
    Q_all = np.array(list((self.jaco_information[p] for p in self.parameter_names))).T
    for i, mea_name in enumerate(self.measurement_variables):
        fim += 1 / self.measurements.variance[str(mea_name)] * (Q_all[i, :].reshape(n, 1) @ Q_all[i, :].reshape(n, 1).T)
    if self.prior_FIM is not None:
        try:
            fim = fim + self.prior_FIM
            self.logger.info('Existed information has been added.')
        except:
            raise ValueError('Check the shape of prior FIM.')
    if np.linalg.cond(fim) > self.max_condition_number:
        self.logger.info('Warning: FIM is near singular. The condition number is: %s ;', np.linalg.cond(fim))
        self.logger.info('A condition number bigger than %s is considered near singular.', self.max_condition_number)
    self._print_FIM_info(fim)
    if self.result is not None:
        self._get_solver_info()
    if self.store_FIM is not None:
        self._store_FIM()