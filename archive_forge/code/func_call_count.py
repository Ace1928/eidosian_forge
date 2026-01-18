import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
@property
def call_count(self):
    return len(self.request_history)