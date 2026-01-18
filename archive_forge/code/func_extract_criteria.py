from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def extract_criteria(self):
    """
        Extract design criteria values for every 'grid' (design variable combination) searched.

        Returns
        -------
        self.store_all_results_dataframe: a pandas dataframe with columns as design variable names and A, D, E, ME-criteria names.
            Each row contains the design variable value for this 'grid', and the 4 design criteria value for this 'grid'.
        """
    store_all_results = []
    search_design_set = product(*self.design_ranges)
    for design_set_iter in search_design_set:
        result_object_asdict = {k: v for k, v in self.FIM_result_list.items() if k == design_set_iter}
        result_object_iter = result_object_asdict[design_set_iter]
        store_iteration_result = list(design_set_iter)
        store_iteration_result.append(result_object_iter.trace)
        store_iteration_result.append(result_object_iter.det)
        store_iteration_result.append(result_object_iter.min_eig)
        store_iteration_result.append(result_object_iter.cond)
        store_all_results.append(store_iteration_result)
    column_names = []
    for i in self.design_names:
        if type(i) is list:
            column_names.append(i[0])
        else:
            column_names.append(i)
    column_names.append('A')
    column_names.append('D')
    column_names.append('E')
    column_names.append('ME')
    store_all_results = np.asarray(store_all_results)
    self.store_all_results_dataframe = pd.DataFrame(store_all_results, columns=column_names)
    if self.store_optimality_name is not None:
        self.store_all_results_dataframe.to_csv(self.store_optimality_name, index=False)