import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def eval_genlaguerre_ldd(n, a, x):
    return eval_genlaguerre(n.astype('l'), a, x)