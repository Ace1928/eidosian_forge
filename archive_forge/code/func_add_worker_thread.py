import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def add_worker_thread(self, *args, **kwargs):
    index = next(self._worker_count)
    worker = threading.Thread(target=self.worker_thread_callback, args=args, kwargs=kwargs, name='worker %d' % index)
    worker.daemon = self.daemon
    worker.start()