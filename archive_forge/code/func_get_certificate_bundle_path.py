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
def get_certificate_bundle_path(cls):
    """:Returns:

        Paths of the .pem encoded certificate to verify connection to
        command executor. Defaults to certifi.where() or
        REQUESTS_CA_BUNDLE env variable if set.
        """
    return cls._ca_certs