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
def custom_showwarning(*args, **kwargs):
    caught_warnings.append(args[0])