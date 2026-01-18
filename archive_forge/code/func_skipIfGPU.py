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
def skipIfGPU(testfn, reason='Test is CPU-only'):
    """
    Decorate a test to skip if a GPU is available.

    Useful for disabling hogwild tests.
    """
    return unittest.skipIf(GPU_AVAILABLE, reason)(testfn)