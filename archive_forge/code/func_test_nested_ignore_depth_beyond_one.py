import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
def test_nested_ignore_depth_beyond_one(self):
    factory = ProgressRecordingUIFactory()
    pb1 = factory.nested_progress_bar()
    pb1.update('foo', 0, 1)
    pb2 = factory.nested_progress_bar()
    pb2.update('foo', 0, 1)
    pb2.finished()
    pb1.finished()
    self.assertEqual([('update', 0, 1, 'foo')], factory._calls)