from io import StringIO
import sys
import dns.exception
import dns.name
import dns.ttl
from ._compat import long, text_type, binary_type
def is_delimiter(self):
    return self.ttype == DELIMITER