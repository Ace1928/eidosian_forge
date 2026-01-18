import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
@property
def encoded_url(self):
    """Access the underlying :class:`URL` object, which has any special
        characters encoded.
        """
    return self._url