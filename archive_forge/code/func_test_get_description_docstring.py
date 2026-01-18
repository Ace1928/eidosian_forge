import argparse
import functools
from cliff import command
from cliff.tests import base
def test_get_description_docstring(self):
    cmd = TestCommand(None, None)
    desc = cmd.get_description()
    assert desc == 'Description of command.\n    '