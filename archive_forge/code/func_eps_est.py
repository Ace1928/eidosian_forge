import pandas as pd
from yfinance.data import YfData
from yfinance.exceptions import YFNotImplementedError
@property
def eps_est(self) -> pd.DataFrame:
    if self._eps_est is None:
        raise YFNotImplementedError('eps_est')
    return self._eps_est