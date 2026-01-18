import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_3str(self):
    options, sources = parse_command_line(['--3str', 'source.pyx'])
    self.assertEqual(options.language_level, '3str')
    self.check_default_global_options()
    self.check_default_options(options, ['language_level'])