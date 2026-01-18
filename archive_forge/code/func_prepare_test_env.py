import sys
import os
import io
import optparse
import re
import unittest
from pprint import pformat
from .test_machinery import PygameTestLoader
def prepare_test_env():
    test_subdir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    main_dir = os.path.split(test_subdir)[0]
    sys.path.insert(0, test_subdir)
    fake_test_subdir = os.path.join(test_subdir, 'run_tests__tests')
    return (main_dir, test_subdir, fake_test_subdir)