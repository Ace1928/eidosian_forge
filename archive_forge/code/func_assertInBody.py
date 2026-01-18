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
def assertInBody(self, value, msg=None):
    """Fail if value not in self.body."""
    if isinstance(value, str):
        value = value.encode(self.encoding)
    if value not in self.body:
        if msg is None:
            msg = '%r not in body: %s' % (value, self.body)
        self._handlewebError(msg)