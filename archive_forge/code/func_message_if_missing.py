from __future__ import print_function
import logging
import socket
import sys
from six.moves import BaseHTTPServer
from six.moves import http_client
from six.moves import input
from six.moves import urllib
from oauth2client import _helpers
from oauth2client import client
def message_if_missing(filename):
    """Helpful message to display if the CLIENT_SECRETS file is missing."""
    return _CLIENT_SECRETS_MESSAGE.format(file_path=filename)