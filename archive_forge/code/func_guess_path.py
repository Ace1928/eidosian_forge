import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
def guess_path(package_root):
    if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
        package_root_base = sys.argv[0][:sys.argv[0].rfind(package_root)]
        binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
        possible_path = os.path.join(package_root_base, package_root, binary)
        logging.info('Guessed test binary path: %s', possible_path)
        if os.access(possible_path, os.X_OK):
            return possible_path
        return None