from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pickle
from itertools import permutations, product
import logging
from enum import Enum
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.sensitivity_toolbox.sens import get_dsdp
from pyomo.contrib.doe.scenario import ScenarioGenerator, FiniteDifferenceStep
from pyomo.contrib.doe.result import FisherResults, GridSearchResult
def compute_FIM(self, mode='direct_kaug', FIM_store_name=None, specified_prior=None, tee_opt=True, scale_nominal_param_value=False, scale_constant_value=1, store_output=None, read_output=None, extract_single_model=None, formula='central', step=0.001):
    """
        This function calculates the Fisher information matrix (FIM) using sensitivity information obtained
        from two possible modes (defined by the CalculationMode Enum):

            1.  sequential_finite: sequentially solve square problems and use finite difference approximation
            2.  direct_kaug: solve a single square problem then extract derivatives using NLP sensitivity theory

        Parameters
        ----------
        mode:
            supports CalculationMode.sequential_finite or CalculationMode.direct_kaug
        FIM_store_name:
            if storing the FIM in a .csv or .txt, give the file name here as a string.
        specified_prior:
            a 2D numpy array providing alternate prior matrix, default is no prior.
        tee_opt:
            if True, IPOPT console output is printed
        scale_nominal_param_value:
            if True, the parameters are scaled by its own nominal value in param_init
        scale_constant_value:
            scale all elements in Jacobian matrix, default is 1.
        store_output:
            if storing the output (value stored in Var 'output_record') as a pickle file, give the file name here as a string.
        read_output:
            if reading the output (value for Var 'output_record') as a pickle file, give the file name here as a string.
        extract_single_model:
            if True, the solved model outputs for each scenario are all recorded as a .csv file.
            The output file uses the name AB.csv, where string A is store_output input, B is the index of scenario.
            scenario index is the number of the scenario outputs which is stored.
        formula:
            choose from the Enum FiniteDifferenceStep.central, .forward, or .backward.
            This option is only used for CalculationMode.sequential_finite mode.
        step:
            Sensitivity perturbation step size, a fraction between [0,1]. default is 0.001

        Returns
        -------
        FIM_analysis: result summary object of this solve
        """
    self.design_values = self.design_vars.variable_names_value
    self.scale_nominal_param_value = scale_nominal_param_value
    self.scale_constant_value = scale_constant_value
    self.formula = FiniteDifferenceStep(formula)
    self.mode = CalculationMode(mode)
    self.step = step
    self.optimize = False
    self.objective_option = ObjectiveLib.zero
    self.tee_opt = tee_opt
    self.FIM_store_name = FIM_store_name
    self.specified_prior = specified_prior
    self.fim_scale_constant_value = self.scale_constant_value ** 2
    square_timer = TicTocTimer()
    square_timer.tic(msg=None)
    if self.mode == CalculationMode.sequential_finite:
        FIM_analysis = self._sequential_finite(read_output, extract_single_model, store_output)
    elif self.mode == CalculationMode.direct_kaug:
        FIM_analysis = self._direct_kaug()
    dT = square_timer.toc(msg=None)
    self.logger.info('elapsed time: %0.1f' % dT)
    return FIM_analysis