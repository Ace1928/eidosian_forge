from nose.plugins.multiprocess import MultiProcessTestRunner  # @UnresolvedImport
from nose.plugins.base import Plugin  # @UnresolvedImport
import sys
from _pydev_runfiles import pydev_runfiles_xml_rpc
import time
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from contextlib import contextmanager
from io import StringIO
import traceback
def new_consolidate(self, result, batch_result):
    """
    Used so that it can work with the multiprocess plugin.
    Monkeypatched because nose seems a bit unsupported at this time (ideally
    the plugin would have this support by default).
    """
    ret = original(self, result, batch_result)
    parent_frame = sys._getframe().f_back
    addr = parent_frame.f_locals['addr']
    i = addr.rindex(':')
    addr = [addr[:i], addr[i + 1:]]
    output, testsRun, failures, errors, errorClasses = batch_result
    if failures or errors:
        for failure in failures:
            PYDEV_NOSE_PLUGIN_SINGLETON.report_cond('fail', addr, output, failure)
        for error in errors:
            PYDEV_NOSE_PLUGIN_SINGLETON.report_cond('error', addr, output, error)
    else:
        PYDEV_NOSE_PLUGIN_SINGLETON.report_cond('ok', addr, output)
    return ret