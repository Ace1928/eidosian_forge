import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def spherical_yn_(n, x):
    return spherical_yn(n.astype('l'), x)