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
def requestJsonAndCheck(self, verb: str, url: str, parameters: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, str]]=None, input: Optional[Any]=None) -> Tuple[Dict[str, Any], Any]:
    return self.__check(*self.requestJson(verb, url, parameters, headers, input, self.__customConnection(url)))