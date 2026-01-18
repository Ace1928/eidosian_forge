from __future__ import annotations
import logging
import sys
import warnings
from logging.config import dictConfig
from types import TracebackType
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union, cast
from twisted.python import log as twisted_log
from twisted.python.failure import Failure
import scrapy
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.settings import Settings
from scrapy.utils.versions import scrapy_components_versions
def get_scrapy_root_handler() -> Optional[logging.Handler]:
    return _scrapy_root_handler