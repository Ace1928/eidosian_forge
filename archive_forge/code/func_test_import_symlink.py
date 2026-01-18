import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_import_symlink(self):
    handler, branch = self.get_handler()
    handler.process(self.get_command_iter(b'foo', 'symlink', b'bar'))