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
def requestMemoryBlobAndCheck(self, verb: str, url: str, parameters: Any, headers: Dict[str, Any], file_like: BinaryIO, cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]]=None) -> Tuple[Dict[str, Any], Any]:

    def encode(_: Any) -> Tuple[str, Any]:
        return (headers['Content-Type'], file_like)
    if not cnx:
        cnx = self.__customConnection(url)
    return self.__check(*self.__requestEncode(cnx, verb, url, parameters, headers, file_like, encode))