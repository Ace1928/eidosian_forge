import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def get_decoded_url(self, lazy=False):
    try:
        return self._decoded_url
    except AttributeError:
        self._decoded_url = DecodedURL(self, lazy=lazy)
    return self._decoded_url