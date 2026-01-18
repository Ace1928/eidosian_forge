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
def skipUnlessTorch14(testfn, reason='Test requires pytorch 1.4+'):
    skip = False
    if not TORCH_AVAILABLE:
        skip = True
    else:
        from packaging import version
        skip = version.parse(torch.__version__) < version.parse('1.4.0')
    return unittest.skipIf(skip, reason)(testfn)