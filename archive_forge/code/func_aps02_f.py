from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps02_f(x):
    """poles at x=n**2, 1st and 2nd derivatives at root are also close to 0"""
    ii = np.arange(1, 21)
    return -2 * np.sum((2 * ii - 5) ** 2 / (x - ii ** 2) ** 3)