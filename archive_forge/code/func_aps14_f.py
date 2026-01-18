from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps14_f(x, n):
    """0 for negative x-values, trigonometric+linear for x positive"""
    if x <= 0:
        return -n / 20.0
    return n / 20.0 * (x / 1.5 + np.sin(x) - 1)