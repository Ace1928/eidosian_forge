from __future__ import print_function
from io import StringIO
import json as _json
import warnings
from typing import Optional, Union
from urllib.parse import quote as urlencode
import pandas as pd
import requests
from . import utils, cache
from .data import YfData
from .scrapers.analysis import Analysis
from .scrapers.fundamentals import Fundamentals
from .scrapers.holders import Holders
from .scrapers.quote import Quote, FastInfo
from .scrapers.history import PriceHistory
from .const import _BASE_URL_, _ROOT_URL_
def get_incomestmt(self, proxy=None, as_dict=False, pretty=False, freq='yearly'):
    return self.get_income_stmt(proxy, as_dict, pretty, freq)