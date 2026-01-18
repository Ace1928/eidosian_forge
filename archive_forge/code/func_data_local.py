import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def data_local(func, dataname, *a, **kw):
    kw.setdefault('dataname', dataname)
    return FuncData(func, DATASETS_LOCAL[dataname], *a, **kw)