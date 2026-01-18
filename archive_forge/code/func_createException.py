import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
@classmethod
def createException(cls, status: int, headers: Dict[str, Any], output: Dict[str, Any]) -> GithubException.GithubException:
    message = output.get('message', '').lower() if output is not None else ''
    exc = GithubException.GithubException
    if status == 401 and message == 'bad credentials':
        exc = GithubException.BadCredentialsException
    elif status == 401 and Consts.headerOTP in headers and re.match('.*required.*', headers[Consts.headerOTP]):
        exc = GithubException.TwoFactorException
    elif status == 403 and message.startswith('missing or invalid user agent string'):
        exc = GithubException.BadUserAgentException
    elif status == 403 and cls.isRateLimitError(message):
        exc = GithubException.RateLimitExceededException
    elif status == 404 and message == 'not found':
        exc = GithubException.UnknownObjectException
    return exc(status, output, headers)