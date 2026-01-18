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
def get_conn(self, auto_open=False):
    """Return a connection to our HTTP server."""
    conn = self._Conn(self.interface(), self.PORT)
    conn.auto_open = auto_open
    conn.connect()
    return conn