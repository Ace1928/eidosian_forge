import sys
import os
import tempfile
import unittest
from ..py3compat import string_types, which
def skip_file_no_x11(name):
    return decorated_dummy(skip_if_no_x11, name) if _x11_skip_cond else None