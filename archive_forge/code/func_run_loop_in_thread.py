from __future__ import print_function
import logging
import os
import socket
import ssl
import sys
import threading
import warnings
from datetime import datetime
import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import trustme
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from urllib3.exceptions import HTTPWarning
from urllib3.util import ALPN_PROTOCOLS, resolve_cert_reqs, resolve_ssl_version
def run_loop_in_thread(io_loop):
    t = threading.Thread(target=io_loop.start)
    t.start()
    return t