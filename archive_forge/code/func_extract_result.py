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
def extract_result(self, anchor: Tag, desc: Tag, hashes: Set) -> Optional[RT]:
    """
        Extracts the result
        """
    link = self.is_valid_link(anchor, hashes)
    if not link:
        return None
    if self.include_title and self.include_description:
        return (link, anchor.text, desc.text)
    if self.include_description:
        return (link, desc.text)
    return (link, anchor.text) if self.include_title else link