import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
@property
def current_search_path(self):
    """The value of ``$PATH`` or :data:`os.defpath` (a string)."""
    return os.environ.get('PATH', os.defpath)