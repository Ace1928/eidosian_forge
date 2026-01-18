import os
import unittest
import contextlib
import tempfile
import shutil
import io
import signal
from typing import Tuple, Dict, Any
from parlai.core.opt import Opt
import parlai.utils.logging as logging
def is_this_circleci():
    """
    Return if we are currently running in CircleCI.
    """
    return bool(os.environ.get('CIRCLECI'))