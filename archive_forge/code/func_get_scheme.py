from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def get_scheme(self):
    """Return the URL scheme being used"""
    return guess_scheme(self.environ)