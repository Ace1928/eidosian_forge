import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
def is_oss():
    """Returns whether the test is run under OSS."""
    return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]