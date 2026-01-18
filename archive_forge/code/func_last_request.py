import urllib.parse
import weakref
from requests.adapters import BaseAdapter
from requests.utils import requote_uri
from requests_mock import exceptions
from requests_mock.request import _RequestObjectProxy
from requests_mock.response import _MatcherResponse
import logging
@property
def last_request(self):
    """Retrieve the latest request sent"""
    try:
        return self.request_history[-1]
    except IndexError:
        return None