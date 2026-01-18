import os
import gzip
import random
import pathlib
import datetime
import contextlib
from http.cookiejar import LWPCookieJar
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, TYPE_CHECKING
def get_random_user_agent():
    """
    Get a random user agent string.

    :rtype: str
    :return: Random user agent string.
    """
    return random.choice(load_user_agents())