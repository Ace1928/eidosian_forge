from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps08_fp(x, n):
    return 2 * x + n * (1 - x) ** (n - 1)