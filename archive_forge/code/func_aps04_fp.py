from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def aps04_fp(x, n, a):
    return n * x ** (n - 1)