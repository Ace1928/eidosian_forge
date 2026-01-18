from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def cplx02_f(z, a):
    """e**z - a: Use to find the log of a"""
    return np.exp(z) - a