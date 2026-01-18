import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_know_valid_values(self):
    self.assertIsTrue('true')
    self.assertIsFalse('false')
    self.assertIsTrue('1')
    self.assertIsFalse('0')
    self.assertIsTrue('on')
    self.assertIsFalse('off')
    self.assertIsTrue('yes')
    self.assertIsFalse('no')
    self.assertIsTrue('y')
    self.assertIsFalse('n')
    self.assertIsTrue('True')
    self.assertIsFalse('False')
    self.assertIsTrue('On')
    self.assertIsFalse('Off')
    self.assertIsTrue('ON')
    self.assertIsFalse('OFF')
    self.assertIsTrue('oN')
    self.assertIsFalse('oFf')