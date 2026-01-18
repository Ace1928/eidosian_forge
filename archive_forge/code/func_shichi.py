import numpy as np
import scipy.special as sc
from scipy.special._testutils import FuncData
def shichi(x):
    shi, chi = sc.shichi(x + 0j)
    return (shi.real, chi.real)