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
def is_cgi(self):
    """Test whether self.path corresponds to a CGI script.

        Returns True and updates the cgi_info attribute to the tuple
        (dir, rest) if self.path requires running a CGI script.
        Returns False otherwise.

        If any exception is raised, the caller should assume that
        self.path was rejected as invalid and act accordingly.

        The default implementation tests whether the normalized url
        path begins with one of the strings in self.cgi_directories
        (and the next character is a '/' or the end of the string).

        """
    collapsed_path = _url_collapse_path(self.path)
    dir_sep = collapsed_path.find('/', 1)
    while dir_sep > 0 and (not collapsed_path[:dir_sep] in self.cgi_directories):
        dir_sep = collapsed_path.find('/', dir_sep + 1)
    if dir_sep > 0:
        head, tail = (collapsed_path[:dir_sep], collapsed_path[dir_sep + 1:])
        self.cgi_info = (head, tail)
        return True
    return False