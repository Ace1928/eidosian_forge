from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps09_fpp(x, n):
    return -12 * n * (1 - n * x) ** 2