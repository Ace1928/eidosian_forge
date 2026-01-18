import abc
import time
import random
import asyncio
import contextlib
from pydantic import BaseModel
from lazyops.imports._aiohttpx import resolve_aiohttpx
from lazyops.imports._bs4 import resolve_bs4
from urllib.parse import quote_plus
import aiohttpx
from bs4 import BeautifulSoup, Tag
from .utils import filter_result, load_user_agents, load_cookie_jar, get_random_jitter
from typing import List, Optional, Dict, Any, Union, Tuple, Set, Callable, Awaitable, TypeVar, Generator, AsyncGenerator
def init_api(self, **kwargs):
    """
        Initializes the api
        """
    if self.api is None:
        self.api = aiohttpx.Client(timeout=self.timeout, cookies=self.cookies, limits=aiohttpx.Limits(max_connections=self.max_connections), proxies={'all://': self.proxy} if self.proxy else None, follow_redirects=True)