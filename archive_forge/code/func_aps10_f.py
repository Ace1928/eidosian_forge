from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps10_f(x, n):
    """Exponential plus a polynomial"""
    return np.exp(-n * x) * (x - 1) + x ** n