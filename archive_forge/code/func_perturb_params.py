import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
def perturb_params(self, vname):
    if self.perturbation_method[vname] == 'gaussian':
        self._perturb_gaussian(vname)
    elif self.perturbation_method[vname] == 'boot':
        self._perturb_bootstrap(vname)
    else:
        raise ValueError('unknown perturbation method')