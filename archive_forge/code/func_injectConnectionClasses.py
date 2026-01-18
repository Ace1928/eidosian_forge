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
def injectConnectionClasses(cls, httpConnectionClass: Type[HTTPRequestsConnectionClass], httpsConnectionClass: Type[HTTPSRequestsConnectionClass]) -> None:
    cls.__persist = False
    cls.__httpConnectionClass = httpConnectionClass
    cls.__httpsConnectionClass = httpsConnectionClass