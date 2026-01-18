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
def requestMultipart(self, verb: str, url: str, parameters: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, Any]]=None, input: Optional[Dict[str, str]]=None, cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]]=None) -> Tuple[int, Dict[str, Any], str]:

    def encode(input: Dict[str, Any]) -> Tuple[str, str]:
        boundary = '----------------------------3c3ba8b523b2'
        eol = '\r\n'
        encoded_input = ''
        for name, value in input.items():
            encoded_input += f'--{boundary}{eol}'
            encoded_input += f'Content-Disposition: form-data; name="{name}"{eol}'
            encoded_input += eol
            encoded_input += value + eol
        encoded_input += f'--{boundary}--{eol}'
        return (f'multipart/form-data; boundary={boundary}', encoded_input)
    return self.__requestEncode(cnx, verb, url, parameters, headers, input, encode)