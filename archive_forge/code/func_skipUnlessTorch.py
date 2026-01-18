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
def skipUnlessTorch(testfn, reason='pytorch is not installed'):
    """
    Decorate a test to skip if torch is not installed.
    """
    return unittest.skipUnless(TORCH_AVAILABLE, reason)(testfn)