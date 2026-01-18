import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def btdtri_comp(a, b, p):
    return btdtri(a, b, 1 - p)