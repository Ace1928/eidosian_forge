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
def openURL(*args, raise_subcls=(), **kwargs):
    """
    Open a URL, retrying when it fails.

    Specify ``raise_subcls`` (class or tuple of classes) to exclude
    those socket.error subclasses from being suppressed and retried.
    """
    opener = functools.partial(_open_url_once, *args, **kwargs)

    def on_exception():
        exc = sys.exc_info()[1]
        if isinstance(exc, raise_subcls):
            raise exc
        time.sleep(0.5)
    return jaraco.functools.retry_call(opener, retries=9, cleanup=on_exception, trap=socket.error)