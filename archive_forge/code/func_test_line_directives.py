import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_line_directives(self):
    options, sources = parse_command_line(['--line-directives', 'source.pyx'])
    self.assertEqual(options.emit_linenums, True)
    self.check_default_global_options()
    self.check_default_options(options, ['emit_linenums'])