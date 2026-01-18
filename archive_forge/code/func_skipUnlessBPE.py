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
def skipUnlessBPE(testfn, reason='Test requires subword NMT'):
    """
    Decorate a test to skip if BPE is not installed.
    """
    return unittest.skipUnless(BPE_INSTALLED, reason)(testfn)