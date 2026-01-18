import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
@classmethod
def reset_timeout(cls):
    """Reset the http request timeout to socket._GLOBAL_DEFAULT_TIMEOUT."""
    cls._timeout = socket._GLOBAL_DEFAULT_TIMEOUT