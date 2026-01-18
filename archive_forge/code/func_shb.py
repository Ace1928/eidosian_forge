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
def shb(response):
    """Return status, headers, body the way we like from a response."""
    resp_status_line = '%s %s' % (response.status, response.reason)
    return (resp_status_line, response.getheaders(), response.read())