import copy
import datetime
import email.utils
import html
import http.client
import io
import itertools
import mimetypes
import os
import posixpath
import select
import shutil
import socket # For gethostbyaddr()
import socketserver
import sys
import time
import urllib.parse
from http import HTTPStatus
def finish_request(self, request, client_address):
    self.RequestHandlerClass(request, client_address, self, directory=args.directory)