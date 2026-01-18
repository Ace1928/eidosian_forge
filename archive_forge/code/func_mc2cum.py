import numpy as np
from scipy.special import comb
def mc2cum(mc):
    """
    just chained because I have still the test case
    """
    first_step = mc2mnc(mc)
    if isinstance(first_step, np.ndarray):
        first_step = first_step.T
    return mnc2cum(first_step)