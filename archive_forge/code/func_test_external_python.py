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
def test_external_python():

    def f(x):
        return 3
    rpy_fun = rinterface.SexpExtPtr.from_pyobject(f)
    _python = rinterface.StrSexpVector(('.Python',))
    res = rinterface.baseenv['.External'](_python, rpy_fun, 1)
    assert len(res) == 1
    assert res[0] == 3