import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_convert_range(self):
    options, sources = parse_command_line(['--convert-range', 'source.pyx'])
    self.assertEqual(Options.convert_range, True)
    self.check_default_global_options(['convert_range'])
    self.check_default_options(options)