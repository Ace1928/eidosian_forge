from .util import FileWrapper, guess_scheme, is_hop_by_hop
from .headers import Headers
import sys, os, time
def result_is_file(self):
    """True if 'self.result' is an instance of 'self.wsgi_file_wrapper'"""
    wrapper = self.wsgi_file_wrapper
    return wrapper is not None and isinstance(self.result, wrapper)