from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import gurobi_path
import os
import sys
from .. import constants
import warnings
def initGurobi(self):
    if self.init_gurobi:
        return
    else:
        self.init_gurobi = True
    try:
        if self.manage_env:
            self.env = gp.Env(params=self.env_options)
            self.model = gp.Model(env=self.env)
        else:
            self.model = gp.Model(env=self.env)
        for param, value in self.solver_params.items():
            self.model.setParam(param, value)
    except gp.GurobiError as e:
        raise e