import re
import sys
import time
def ne_(a, b, msg=None):
    """Assert a != b, with repr messaging on failure."""
    assert a != b, msg or '%r == %r' % (a, b)