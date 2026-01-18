import sys
import unittest
from unittest import mock
from bpython.curtsiesfrontend.coderunner import CodeRunner, FakeOutput
def ctrlc():
    raise KeyboardInterrupt()