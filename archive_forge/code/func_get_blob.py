import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def get_blob(self, sha):
    """Return the blob named sha from the test data dir."""
    return self.get_sha_file(Blob, 'blobs', sha)