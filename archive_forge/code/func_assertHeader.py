from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def assertHeader(self, key, value=None, msg=None):
    """Fail if (key, [value]) not in self.headers."""
    lowkey = key.lower()
    for k, v in self.headers:
        if k.lower() == lowkey:
            if value is None or str(value) == v:
                return v
    if msg is None:
        if value is None:
            msg = '%r not in headers' % key
        else:
            msg = '%r:%r not in headers' % (key, value)
    self._handlewebError(msg)