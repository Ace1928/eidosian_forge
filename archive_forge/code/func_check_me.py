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
def check_me(self, obj: 'GithubObject') -> None:
    if self.DEBUG_FLAG and self.ON_CHECK_ME is not None:
        frame = None
        if self.DEBUG_HEADER_KEY in obj._headers:
            frame_index = obj._headers[self.DEBUG_HEADER_KEY]
            frame = self._frameBuffer[frame_index]
        self.ON_CHECK_ME(obj, frame)