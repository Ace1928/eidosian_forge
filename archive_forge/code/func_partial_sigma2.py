from statsmodels.compat.pandas import MONTH_END
import os
import re
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import nile
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.mlemodel import MLEModel, MLEResultsWrapper
from statsmodels.tsa.statespace.tests.results import (
def partial_sigma2(phi, sigma2):
    return -0.5 * (2 * sigma2 + phi - 1.25) / sigma2 ** 2