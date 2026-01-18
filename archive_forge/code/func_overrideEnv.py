import doctest
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import ClassVar, List
from unittest import SkipTest, expectedFailure, skipIf
from unittest import TestCase as _TestCase
def overrideEnv(name, value):
    oldval = os.environ.get(name)
    if value is not None:
        os.environ[name] = value
    else:
        del os.environ[name]
    to_restore.append((name, oldval))