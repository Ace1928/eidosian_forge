import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def functest_builder(method, func):
    """Generate a test method that tests the given function."""

    def do_test(self):
        method(self, func)
    return do_test