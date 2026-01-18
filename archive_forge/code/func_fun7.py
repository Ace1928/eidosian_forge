from random import random
import numpy as np
from scipy.optimize import _zeros_py as cc
def fun7(x):
    return 0 if abs(x) < 0.00038 else x * np.exp(-x ** (-2))