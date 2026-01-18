import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def ellipeinc_(f, k):
    return ellipeinc(f, k * k)