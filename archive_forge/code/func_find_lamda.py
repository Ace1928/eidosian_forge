import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def find_lamda(upperlimit, Gbar, b, radius):
    lowerlimit = upperlimit
    step = 0.1
    while f(lowerlimit, Gbar, b, radius) < 0:
        lowerlimit -= step
    converged = False
    while not converged:
        midt = (upperlimit + lowerlimit) / 2.0
        lamda = midt
        fmidt = f(midt, Gbar, b, radius)
        fupper = f(upperlimit, Gbar, b, radius)
        if fupper * fmidt < 0:
            lowerlimit = midt
        else:
            upperlimit = midt
        if abs(upperlimit - lowerlimit) < 1e-06:
            converged = True
    return lamda