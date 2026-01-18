import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_format_delta(self):
    self.assertFormatedDelta('0 seconds ago', 0)
    self.assertFormatedDelta('1 second ago', 1)
    self.assertFormatedDelta('10 seconds ago', 10)
    self.assertFormatedDelta('59 seconds ago', 59)
    self.assertFormatedDelta('89 seconds ago', 89)
    self.assertFormatedDelta('1 minute, 30 seconds ago', 90)
    self.assertFormatedDelta('3 minutes, 0 seconds ago', 180)
    self.assertFormatedDelta('3 minutes, 1 second ago', 181)
    self.assertFormatedDelta('10 minutes, 15 seconds ago', 615)
    self.assertFormatedDelta('30 minutes, 59 seconds ago', 1859)
    self.assertFormatedDelta('31 minutes, 0 seconds ago', 1860)
    self.assertFormatedDelta('60 minutes, 0 seconds ago', 3600)
    self.assertFormatedDelta('89 minutes, 59 seconds ago', 5399)
    self.assertFormatedDelta('1 hour, 30 minutes ago', 5400)
    self.assertFormatedDelta('2 hours, 30 minutes ago', 9017)
    self.assertFormatedDelta('10 hours, 0 minutes ago', 36000)
    self.assertFormatedDelta('24 hours, 0 minutes ago', 86400)
    self.assertFormatedDelta('35 hours, 59 minutes ago', 129599)
    self.assertFormatedDelta('36 hours, 0 minutes ago', 129600)
    self.assertFormatedDelta('36 hours, 0 minutes ago', 129601)
    self.assertFormatedDelta('36 hours, 1 minute ago', 129660)
    self.assertFormatedDelta('36 hours, 1 minute ago', 129661)
    self.assertFormatedDelta('84 hours, 10 minutes ago', 303002)
    self.assertFormatedDelta('84 hours, 10 minutes in the future', -303002)
    self.assertFormatedDelta('1 second in the future', -1)
    self.assertFormatedDelta('2 seconds in the future', -2)