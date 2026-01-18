import os
import gzip
import random
import pathlib
import datetime
import contextlib
from http.cookiejar import LWPCookieJar
from urllib.parse import urlparse, parse_qs
from typing import List, Optional, TYPE_CHECKING
def load_user_agents():
    """
    Loads the user agents from the user agents file.
    """
    global _user_agents
    if _user_agents is None:
        user_agents_file = lib_path.joinpath('user_agents.txt.gz')
        fp = gzip.open(user_agents_file.as_posix(), 'rb')
        try:
            _user_agents = [_.decode().strip() for _ in fp.readlines()]
        except Exception as e:
            _user_agents = [USER_AGENT]
        finally:
            fp.close()
    return _user_agents