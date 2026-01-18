import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def sph_harm_(m, n, theta, phi):
    y = sph_harm(m, n, theta, phi)
    return (y.real, y.imag)