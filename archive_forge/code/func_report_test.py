from _pydev_runfiles import pydev_runfiles_xml_rpc
import pickle
import zlib
import base64
import os
from pydevd_file_utils import canonical_normalized_path
import pytest
import sys
import time
from pathlib import Path
def report_test(status, filename, test, captured_output, error_contents, duration):
    """
    @param filename: 'D:\\src\\mod1\\hello.py'
    @param test: 'TestCase.testMet1'
    @param status: fail, error, ok
    """
    time_str = '%.2f' % (duration,)
    pydev_runfiles_xml_rpc.notifyTest(status, captured_output, error_contents, filename, test, time_str)