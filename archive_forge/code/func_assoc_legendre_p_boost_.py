import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def assoc_legendre_p_boost_(nu, mu, x):
    return lpmv(mu, nu.astype(int), x)