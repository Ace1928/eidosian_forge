import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def mathieu_ms1_scaled(m, q, x):
    return mathieu_modsem1(m, q, x)[0] * np.sqrt(np.pi / 2)