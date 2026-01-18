import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def init_mpl_modules_for_patching(self):
    from pydev_ipython.matplotlibtools import activate_matplotlib, activate_pylab, activate_pyplot
    self.mpl_modules_for_patching = {'matplotlib': lambda: activate_matplotlib(self.enableGui), 'matplotlib.pyplot': activate_pyplot, 'pylab': activate_pylab}