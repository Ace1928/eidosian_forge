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
def noheaders():
    """Return an empty email Message object."""
    global _noheaders
    if _noheaders is None:
        _noheaders = email.message_from_string('')
    return _noheaders