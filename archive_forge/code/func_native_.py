import sys
import types
from cgi import parse_header
def native_(s, encoding='latin-1', errors='strict'):
    if isinstance(s, text_type):
        return s.encode(encoding, errors)
    return str(s)