import os
import gzip
import random
import pathlib
import datetime
import contextlib
from http.cookiejar import LWPCookieJar
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, TYPE_CHECKING
def get_random_jitter(min_seconds: int=2, max_seconds: int=5) -> int:
    """
    Gets a random jitter
    """
    return random.randint(min_seconds, max_seconds)