import sys
import os
import tempfile
import unittest
from ..py3compat import string_types, which
def skipper_func(*args, **kwargs):
    """Skipper for normal test functions."""
    if skip_val():
        raise nose.SkipTest(get_msg(f, msg))
    else:
        return f(*args, **kwargs)