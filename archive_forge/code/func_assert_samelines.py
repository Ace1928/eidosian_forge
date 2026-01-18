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
def assert_samelines(testcase: TestCase, text1: str, text2: str, msg: Optional[str]=None) -> None:
    """Asserts text1 and text2 have the same lines, ignoring differences in
    line endings between platforms
    """
    testcase.assertEqual(text1.splitlines(), text2.splitlines(), msg)