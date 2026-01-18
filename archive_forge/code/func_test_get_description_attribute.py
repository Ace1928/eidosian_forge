import argparse
import functools
from cliff import command
from cliff.tests import base
def test_get_description_attribute(self):
    cmd = TestCommand(None, None)
    cmd._description = 'this is not the default'
    desc = cmd.get_description()
    assert desc == 'this is not the default'