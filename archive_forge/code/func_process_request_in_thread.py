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
def process_request_in_thread(self, request, client_address):
    """
        The worker thread should call back here to do the rest of the
        request processing. Error handling normaller done in 'handle_request'
        must be done here.
        """
    try:
        self.finish_request(request, client_address)
        self.close_request(request)
    except BaseException as e:
        self.handle_error(request, client_address)
        self.close_request(request)
        if isinstance(e, (MemoryError, KeyboardInterrupt)):
            raise