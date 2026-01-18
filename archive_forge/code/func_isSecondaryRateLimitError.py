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
def isSecondaryRateLimitError(cls, message: str) -> bool:
    if not message:
        return False
    message = message.lower()
    return message.startswith('you have exceeded a secondary rate limit') or message.endswith('please retry your request again later.') or message.endswith('please wait a few minutes before you try again.')