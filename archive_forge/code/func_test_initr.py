import gc
import multiprocessing
import os
import pickle
import pytest
from rpy2 import rinterface
import rpy2
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import signal
import sys
import subprocess
import tempfile
import textwrap
import time
def test_initr():
    preserve_hash = True
    args = ()
    if os.name != 'nt':
        args = (preserve_hash,)
    proc = multiprocessing.Process(target=_init_r, args=args)
    proc.start()
    proc.join()