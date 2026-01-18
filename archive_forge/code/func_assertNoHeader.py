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
def assertNoHeader(self, key, msg=None):
    """Fail if key in self.headers."""
    lowkey = key.lower()
    matches = [k for k, v in self.headers if k.lower() == lowkey]
    if matches:
        if msg is None:
            msg = '%r in headers' % key
        self._handlewebError(msg)