import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def assertIsTrue(self, s, accepted_values=None):
    res = _mod_ui.bool_from_string(s, accepted_values=accepted_values)
    self.assertEqual(True, res)