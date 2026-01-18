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
def test_get_initoptions():
    options = rinterface.embedded.get_initoptions()
    assert len(rinterface.embedded._options) == len(options)
    for o1, o2 in zip(rinterface.embedded._options, options):
        assert o1 == o2