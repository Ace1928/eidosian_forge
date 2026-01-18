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
def test_callback():
    callbacklist = []

    def callback():
        callbacklist.append(1)
    with utils.obj_in_module(callbacks, 'callback', callback):
        callbacks._callback()
    assert tuple(callbacklist) == (1,)