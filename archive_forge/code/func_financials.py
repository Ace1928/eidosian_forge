import datetime
import json
import pandas as pd
from yfinance import utils, const
from yfinance.data import YfData
from yfinance.exceptions import YFinanceException, YFNotImplementedError
@property
def financials(self) -> 'Financials':
    return self._financials