import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
@property
def earnings(self) -> dict:
    if self._earnings is None:
        raise YFNotImplementedError('earnings')
    return self._earnings