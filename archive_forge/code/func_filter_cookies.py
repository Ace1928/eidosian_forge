import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sized
from http.cookies import BaseCookie, Morsel
from typing import (
from multidict import CIMultiDict
from yarl import URL
from .helpers import get_running_loop
from .typedefs import LooseCookies
@abstractmethod
def filter_cookies(self, request_url: URL) -> 'BaseCookie[str]':
    """Return the jar's cookies filtered by their attributes."""