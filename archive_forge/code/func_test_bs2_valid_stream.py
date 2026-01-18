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
def test_bs2_valid_stream(self):
    """
        Test --datatype valid:stream.
        """
    return self._run_display_data('valid:stream', batchsize=2)