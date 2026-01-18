import pyomo.environ as pyo
from pyomo.contrib.pynumero.examples.callback.reactor_design import model as m
from pyomo.common.dependencies import pandas as pd
def get_residual_dataframe(self):
    return pd.DataFrame(self._residuals)