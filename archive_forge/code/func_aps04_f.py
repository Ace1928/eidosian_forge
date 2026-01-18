from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps04_f(x, n, a):
    """Medium-degree polynomial"""
    return x ** n - a