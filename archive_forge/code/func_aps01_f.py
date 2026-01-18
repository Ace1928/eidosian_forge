from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps01_f(x):
    """Straightforward sum of trigonometric function and polynomial"""
    return np.sin(x) - x / 2