import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_Werror(self):
    options, sources = parse_command_line(['-Werror', 'source.pyx'])
    self.assertEqual(Options.warning_errors, True)
    self.check_default_global_options(['warning_errors'])
    self.check_default_options(options)