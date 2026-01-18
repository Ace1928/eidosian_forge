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
def requestMultipartAndCheck(self, verb: str, url: str, parameters: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None, input: Optional[Dict[str, str]]=None) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    return self.__check(*self.requestMultipart(verb, url, parameters, headers, input, self.__customConnection(url)))