import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
def import_devtools(ver):
    """Attempt to load the current latest available devtools into the module
    cache for use later."""
    global devtools
    global version
    version = ver
    base = 'selenium.webdriver.common.devtools.v'
    try:
        devtools = importlib.import_module(f'{base}{ver}')
        return devtools
    except ModuleNotFoundError:
        devtools_path = pathlib.Path(__file__).parents[1].joinpath('devtools')
        versions = tuple((f.name for f in devtools_path.iterdir() if f.is_dir()))
        latest = max((int(x[1:]) for x in versions))
        selenium_logger = logging.getLogger(__name__)
        selenium_logger.debug('Falling back to loading `devtools`: v%s', latest)
        devtools = importlib.import_module(f'{base}{latest}')
        return devtools