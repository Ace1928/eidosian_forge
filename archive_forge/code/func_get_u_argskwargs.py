from scipy import stats
from scipy.stats import distributions
import numpy as np
def get_u_argskwargs(**kwargs):
    u_kwargs = {k.replace('u_', '', 1): v for k, v in kwargs.items() if k.startswith('u_')}
    u_args = u_kwargs.pop('u_args', None)
    return (u_args, u_kwargs)