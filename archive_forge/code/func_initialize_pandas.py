from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from pandas import Panel
def initialize_pandas(self, panel_data, endog_name, exog_name):
    self.panel_data = panel_data
    endog = panel_data[endog_name].values
    self.endog = np.squeeze(endog)
    if exog_name is None:
        exog_name = panel_data.columns.tolist()
        exog_name.remove(endog_name)
    self.exog = panel_data.filterItems(exog_name).values
    self._exog_name = exog_name
    self._endog_name = endog_name
    self._timeseries = panel_data.major_axis
    self._panelseries = panel_data.minor_axis