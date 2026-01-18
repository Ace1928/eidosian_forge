import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def mathieu_se_rad(m, q, x):
    return mathieu_sem(m, q, x * 180 / np.pi)[0]