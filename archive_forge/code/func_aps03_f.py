from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps03_f(x, a, b):
    """Rapidly changing at the root"""
    return a * x * np.exp(b * x)