from __future__ import print_function
import datetime as _datetime
import logging
import re as _re
import sys as _sys
import threading
from functools import lru_cache
from inspect import getmembers
from types import FunctionType
from typing import List, Optional
import numpy as _np
import pandas as _pd
import pytz as _tz
import requests as _requests
from dateutil.relativedelta import relativedelta
from pytz import UnknownTimeZoneError
from yfinance import const
from .const import _BASE_URL_
def build_template(data):
    """
    build_template returns the details required to rebuild any of the yahoo finance financial statements in the same order as the yahoo finance webpage. The function is built to be used on the "FinancialTemplateStore" json which appears in any one of the three yahoo finance webpages: "/financials", "/cash-flow" and "/balance-sheet".

    Returns:
        - template_annual_order: The order that annual figures should be listed in.
        - template_ttm_order: The order that TTM (Trailing Twelve Month) figures should be listed in.
        - template_order: The order that quarterlies should be in (note that quarterlies have no pre-fix - hence why this is required).
        - level_detail: The level of each individual line item. E.g. for the "/financials" webpage, "Total Revenue" is a level 0 item and is the summation of "Operating Revenue" and "Excise Taxes" which are level 1 items.

    """
    template_ttm_order = []
    template_annual_order = []
    template_order = []
    level_detail = []

    def traverse(node, level):
        """
        A recursive function that visits a node and its children.

        Args:
            node: The current node in the data structure.
            level: The depth of the current node in the data structure.
        """
        if level > 5:
            return
        template_ttm_order.append(f'trailing{node['key']}')
        template_annual_order.append(f'annual{node['key']}')
        template_order.append(f'{node['key']}')
        level_detail.append(level)
        if 'children' in node:
            for child in node['children']:
                traverse(child, level + 1)
    for key in data['template']:
        traverse(key, 0)
    return (template_ttm_order, template_annual_order, template_order, level_detail)