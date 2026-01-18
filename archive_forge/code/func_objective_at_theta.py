import re
import importlib as im
import logging
import types
import json
from itertools import combinations
from pyomo.common.dependencies import (
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.environ import Block, ComponentUID
import pyomo.contrib.parmest.utils as utils
import pyomo.contrib.parmest.graphics as graphics
from pyomo.dae import ContinuousSet
def objective_at_theta(self, theta_values=None, initialize_parmest_model=False):
    """
        Objective value for each theta

        Parameters
        ----------
        theta_values: pd.DataFrame, columns=theta_names
            Values of theta used to compute the objective

        initialize_parmest_model: boolean
            If True: Solve square problem instance, build extensive form of the model for
            parameter estimation, and set flag model_initialized to True


        Returns
        -------
        obj_at_theta: pd.DataFrame
            Objective value for each theta (infeasible solutions are
            omitted).
        """
    if len(self.theta_names) == 1 and self.theta_names[0] == 'parmest_dummy_var':
        pass
    else:
        model_temp = self._create_parmest_model(self.callback_data[0])
        model_theta_list = []
        for theta_i in self.theta_names:
            var_cuid = ComponentUID(theta_i)
            var_validate = var_cuid.find_component_on(model_temp)
            try:
                set_cuid = ComponentUID(var_validate.index_set())
                set_validate = set_cuid.find_component_on(model_temp)
                for s in set_validate:
                    self_theta_temp = repr(var_cuid) + '[' + repr(s) + ']'
                    model_theta_list.append(self_theta_temp)
            except AttributeError:
                self_theta_temp = repr(var_cuid)
                model_theta_list.append(self_theta_temp)
            except:
                raise
        if set(self.theta_names) == set(model_theta_list) and len(self.theta_names) == set(model_theta_list):
            pass
        else:
            self.theta_names_updated = model_theta_list
    if theta_values is None:
        all_thetas = {}
        theta_names = self._return_theta_names()
    else:
        assert isinstance(theta_values, pd.DataFrame)
        theta_names = theta_values.columns
        for theta in list(theta_names):
            theta_temp = theta.replace("'", '')
            assert theta_temp in [t.replace("'", '') for t in model_theta_list], "Theta name {} in 'theta_values' not in 'theta_names' {}".format(theta_temp, model_theta_list)
        assert len(list(theta_names)) == len(model_theta_list)
        all_thetas = theta_values.to_dict('records')
    if all_thetas:
        task_mgr = utils.ParallelTaskManager(len(all_thetas))
        local_thetas = task_mgr.global_to_local_data(all_thetas)
    elif initialize_parmest_model:
        task_mgr = utils.ParallelTaskManager(1)
    all_obj = list()
    if len(all_thetas) > 0:
        for Theta in local_thetas:
            obj, thetvals, worststatus = self._Q_at_theta(Theta, initialize_parmest_model=initialize_parmest_model)
            if worststatus != pyo.TerminationCondition.infeasible:
                all_obj.append(list(Theta.values()) + [obj])
    else:
        obj, thetvals, worststatus = self._Q_at_theta(thetavals={}, initialize_parmest_model=initialize_parmest_model)
        if worststatus != pyo.TerminationCondition.infeasible:
            all_obj.append(list(thetvals.values()) + [obj])
    global_all_obj = task_mgr.allgather_global_data(all_obj)
    dfcols = list(theta_names) + ['obj']
    obj_at_theta = pd.DataFrame(data=global_all_obj, columns=dfcols)
    return obj_at_theta