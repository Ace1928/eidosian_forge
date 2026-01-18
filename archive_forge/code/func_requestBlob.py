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
def requestBlob(self, verb: str, url: str, parameters: Optional[Dict[str, str]]=None, headers: Optional[Dict[str, str]]=None, input: Optional[str]=None, cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]]=None) -> Tuple[int, Dict[str, Any], str]:
    if headers is None:
        headers = {}

    def encode(local_path: str) -> Tuple[str, Any]:
        if 'Content-Type' in headers:
            mime_type = headers['Content-Type']
        else:
            guessed_type = mimetypes.guess_type(local_path)
            mime_type = guessed_type[0] if guessed_type[0] is not None else Consts.defaultMediaType
        f = open(local_path, 'rb')
        return (mime_type, f)
    if input:
        headers['Content-Length'] = str(os.path.getsize(input))
    return self.__requestEncode(cnx, verb, url, parameters, headers, input, encode)