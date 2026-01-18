from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys
def run_tests(self):
    import pytest
    errno = pytest.main(self.test_args)
    sys.exit(errno)