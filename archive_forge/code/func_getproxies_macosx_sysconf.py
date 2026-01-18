import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def getproxies_macosx_sysconf():
    """Return a dictionary of scheme -> proxy server URL mappings.

        This function uses the MacOSX framework SystemConfiguration
        to fetch the proxy information.
        """
    return _get_proxies()