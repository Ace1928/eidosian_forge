import numpy as np
from scipy import stats
import pandas as pd
@property
def se_obs(self):
    return np.sqrt(self.var_pred_mean + self.var_resid)