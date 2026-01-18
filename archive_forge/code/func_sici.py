import numpy as np
import scipy.special as sc
from scipy.special._testutils import FuncData
def sici(x):
    si, ci = sc.sici(x + 0j)
    return (si.real, ci.real)