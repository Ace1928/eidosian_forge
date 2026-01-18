import pandas as pd
from yfinance.data import YfData
from yfinance.exceptions import YFNotImplementedError
@property
def analyst_price_target(self) -> pd.DataFrame:
    if self._analyst_price_target is None:
        raise YFNotImplementedError('analyst_price_target')
    return self._analyst_price_target