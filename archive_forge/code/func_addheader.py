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
def addheader(self, *args):
    """Add a header to be used by the HTTP interface only
        e.g. u.addheader('Accept', 'sound/basic')"""
    self.addheaders.append(args)