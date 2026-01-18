import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def eval_legendre_ld(n, x):
    return eval_legendre(n.astype('l'), x)