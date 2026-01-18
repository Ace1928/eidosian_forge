import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def module_cleanup1(*args, **kwargs):
    module_cleanups.append((3, args, kwargs))