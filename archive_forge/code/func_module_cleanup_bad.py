import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def module_cleanup_bad(*args, **kwargs):
    raise CustomError('CleanUpExc')