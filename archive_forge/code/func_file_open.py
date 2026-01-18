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
def file_open(self, req):
    url = req.selector
    if url[:2] == '//' and url[2:3] != '/' and (req.host and req.host != 'localhost'):
        if not req.host in self.get_names():
            raise URLError('file:// scheme is supported only on localhost')
    else:
        return self.open_local_file(req)