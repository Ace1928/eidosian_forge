import os
from base64 import b64encode
from collections import deque
from http.client import HTTPConnection
from json import loads
from threading import Event, Thread
from time import sleep
from urllib.parse import urlparse, urlunparse
import requests
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.weakmethod import WeakMethod
def call_request(self, body, headers):
    timeout = self._timeout
    ca_file = self.ca_file
    verify = self.verify
    url = self._requested_url
    auth = self._auth
    req = requests
    kwargs = {}
    if self._method is None:
        method = 'get' if body is None else 'post'
    else:
        method = self._method.lower()
    req_call = getattr(req, method)
    if auth:
        kwargs['auth'] = auth
    response = req_call(url, data=body, headers=headers, timeout=timeout, verify=verify, cert=ca_file, **kwargs)
    return (None, response)