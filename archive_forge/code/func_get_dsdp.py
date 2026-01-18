from pyomo.environ import (
from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
def get_dsdp(model, theta_names, theta, tee=False):
    """This function calculates gradient vector of the variables
        with respect to the parameters (theta_names).

    e.g) min f:  p1*x1+ p2*(x2^2) + p1*p2
         s.t  c1: x1 + x2 = p1
              c2: x2 + x3 = p2
              0 <= x1, x2, x3 <= 10
              p1 = 10
              p2 = 5
    the function returns dx/dp and dp/dp, and column orders.

    The following terms are used to define the output dimensions:
    Ncon   = number of constraints
    Nvar   = number of variables (Nx + Ntheta)
    Nx     = number of decision (primal) variables
    Ntheta = number of uncertain parameters.

    Parameters
    ----------
    model: Pyomo ConcreteModel
        model should include an objective function
    theta_names: list of strings
        List of Var names
    theta: dict
        Estimated parameters e.g) from parmest
    tee: bool, optional
        Indicates that ef solver output should be teed

    Returns
    -------
    dsdp: scipy.sparse.csr.csr_matrix
        Ntheta by Nvar size sparse matrix. A Jacobian matrix of the
        (decision variables, parameters) with respect to parameters
        (theta_names). number of rows = len(theta_name), number of
        columns = len(col)
    col: list
        List of variable names
    """
    param_list = []
    for name in theta_names:
        comp = model.find_component(name)
        if comp is None:
            raise RuntimeError('Cannot find component %s on model' % name)
        if comp.ctype is Var:
            comp.fix()
        param_list.append(comp)
    sens = SensitivityInterface(model, clone_model=True)
    m = sens.model_instance
    sens.setup_sensitivity(param_list)
    k_aug = K_augInterface()
    k_aug.k_aug(m, tee=tee)
    nl_data = {}
    with InTempDir():
        base_fname = 'col_row'
        nl_file = '.'.join((base_fname, 'nl'))
        row_file = '.'.join((base_fname, 'row'))
        col_file = '.'.join((base_fname, 'col'))
        m.write(nl_file, io_options={'symbolic_solver_labels': True})
        for fname in [nl_file, row_file, col_file]:
            with open(fname, 'r') as fp:
                nl_data[fname] = fp.read()
    dsdp = np.fromstring(k_aug.data['dsdp_in_.in'], sep='\n\t')
    col = nl_data[col_file].strip('\n').split('\n')
    row = nl_data[row_file].strip('\n').split('\n')
    dsdp = dsdp.reshape((len(theta_names), int(len(dsdp) / len(theta_names))))
    dsdp = dsdp[:len(theta_names), :len(col)]
    col = [i for i in col if sens.get_default_block_name() not in i]
    dsdp_out = np.zeros((len(theta_names), len(col)))
    for i in range(len(theta_names)):
        for j in range(len(col)):
            if sens.get_default_block_name() not in col[j]:
                dsdp_out[i, j] = -dsdp[i, j]
    return (scipy.sparse.csr_matrix(dsdp_out), col)