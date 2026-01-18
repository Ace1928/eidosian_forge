import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
def set_hundred_continue_response_headers(self, headers, capitalize_response_headers=True):
    if capitalize_response_headers:
        headers = [('-'.join([x.capitalize() for x in key.split('-')]), value) for key, value in headers]
    self.hundred_continue_headers = headers