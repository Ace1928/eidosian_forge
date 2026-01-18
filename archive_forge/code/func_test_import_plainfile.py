import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_import_plainfile(self):
    handler, branch = self.get_handler()
    handler.process(self.get_command_iter(b'foo', 'file', b'aaa'))