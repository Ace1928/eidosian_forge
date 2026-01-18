import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def fakeRegisterResult(thisResult):
    self.wasRegistered += 1
    self.assertEqual(thisResult, result)