from .. import utils
import builtins
import io
import logging
import os
import pytest
import tempfile
import sys
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
def test_busy():
    busylist = []

    def busy(which):
        busylist.append(which)
    with utils.obj_in_module(callbacks, 'busy', busy):
        which = 1
        callbacks._busy(which)
    assert tuple(busylist) == (1,)