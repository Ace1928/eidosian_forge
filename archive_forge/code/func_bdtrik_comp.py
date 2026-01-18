import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def bdtrik_comp(y, n, p):
    return bdtrik(1 - y, n, p)