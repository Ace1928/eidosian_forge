import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def test_dump_options(self):
    msgs = []

    def _announce(msg, level):
        msgs.append(msg)
    cmd = self.cmd
    cmd.announce = _announce
    cmd.option1 = 1
    cmd.option2 = 1
    cmd.user_options = [('option1', '', ''), ('option2', '', '')]
    cmd.dump_options()
    wanted = ["command options for 'MyCmd':", '  option1 = 1', '  option2 = 1']
    self.assertEqual(msgs, wanted)