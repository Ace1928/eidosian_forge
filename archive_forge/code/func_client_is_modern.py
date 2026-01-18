from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def client_is_modern(self):
    """True if client can accept status and headers"""
    return self.environ['SERVER_PROTOCOL'].upper() != 'HTTP/0.9'