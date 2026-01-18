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
def ip2num(ipAddr):
    parts = ipAddr.split('.')
    parts = list(map(int, parts))
    if len(parts) != 4:
        parts = (parts + [0, 0, 0, 0])[:4]
    return parts[0] << 24 | parts[1] << 16 | parts[2] << 8 | parts[3]