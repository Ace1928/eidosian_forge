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
def setup_warning_catcher():
    """Wrap warnings.showwarning with code that records warnings."""
    caught_warnings = []
    original_showwarning = warnings.showwarning

    def custom_showwarning(*args, **kwargs):
        caught_warnings.append(args[0])
    warnings.showwarning = custom_showwarning

    def restore_showwarning():
        warnings.showwarning = original_showwarning
    return (caught_warnings, restore_showwarning)