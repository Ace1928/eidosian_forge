from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps01_fp(x):
    return np.cos(x) - 1.0 / 2