from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps03_fpp(x, a, b):
    return a * (b * (b * x + 1) + b) * np.exp(b * x)