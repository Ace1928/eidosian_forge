import os
from pyomo.environ import SolverFactory
from pyomo.common.tempfiles import TempfileManager
def set_k_aug_options(self, **options):
    for key, val in options.items():
        self._k_aug.options[key] = val