import asyncio
import os
from importlib import import_module
from pathlib import Path
from posixpath import split
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type
from unittest import TestCase, mock
from twisted.internet.defer import Deferred
from twisted.trial.unittest import SkipTest
from scrapy import Spider
from scrapy.crawler import Crawler
from scrapy.utils.boto import is_botocore_available
def buffer_data(data: bytes) -> None:
    ftp_data.append(data)