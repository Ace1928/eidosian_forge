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
def is_AQUA_or_Windows(function):
    platform = rinterface.baseenv.find('.Platform')
    names = platform.do_slot('names')
    platform_gui = names[names.index('GUI')]
    platform_ostype = names[names.index('OS.type')]
    if platform_gui != 'AQUA' and platform_ostype != 'windows':
        return False
    else:
        return True