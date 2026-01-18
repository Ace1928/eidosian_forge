import re
import sys
from functools import wraps
from inspect import getmembers
from types import CoroutineType
from typing import AsyncGenerator, Dict
from unittest import TestCase
from scrapy.http import Request
from scrapy.utils.python import get_spec
from scrapy.utils.spider import iterate_spider_output
@wraps(cb)
def cb_wrapper(response, **cb_kwargs):
    try:
        output = cb(response, **cb_kwargs)
        output = list(iterate_spider_output(output))
    except Exception:
        case = _create_testcase(method, 'callback')
        results.addError(case, sys.exc_info())