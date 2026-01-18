import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_invalid_values(self):
    self.assertIsNone(None)
    self.assertIsNone('doubt')
    self.assertIsNone('frue')
    self.assertIsNone('talse')
    self.assertIsNone('42')