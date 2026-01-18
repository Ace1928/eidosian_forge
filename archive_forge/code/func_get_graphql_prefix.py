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
@staticmethod
def get_graphql_prefix(path: Optional[str]) -> str:
    if path is None or path in ['', '/']:
        path = ''
    if path.endswith(('/v3', '/v3/')):
        path = Requester.remove_suffix(path, '/')
        path = Requester.remove_suffix(path, '/v3')
    return path + '/graphql'