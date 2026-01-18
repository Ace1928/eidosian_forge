import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_no_c_in_traceback(self):
    options, sources = parse_command_line(['--no-c-in-traceback', 'source.pyx'])
    self.assertEqual(options.c_line_in_traceback, False)
    self.check_default_global_options()
    self.check_default_options(options, ['c_line_in_traceback'])