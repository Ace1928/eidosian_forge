from statsmodels.sandbox.predict_functional import predict_functional
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
def pctl(q):
    return lambda x: np.percentile(x, 100 * q)