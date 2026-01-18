from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps11_f(x, n):
    """Rational function with a zero at x=1/n and a pole at x=0"""
    return (n * x - 1) / ((n - 1) * x)