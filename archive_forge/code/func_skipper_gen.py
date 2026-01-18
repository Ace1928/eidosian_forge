import sys
import os
import tempfile
import unittest
from ..py3compat import string_types, which
def skipper_gen(*args, **kwargs):
    """Skipper for test generators."""
    if skip_val():
        raise nose.SkipTest(get_msg(f, msg))
    else:
        for x in f(*args, **kwargs):
            yield x