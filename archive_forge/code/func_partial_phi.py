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
def partial_phi(phi, sigma2):
    return -0.5 * (phi ** 2 + 2 * phi * sigma2 - 1) / (sigma2 * (1 - phi ** 2))