import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_gdb_first(self):
    options, sources = parse_command_line(['--gdb', '--gdb-outdir=my_dir', 'file3.pyx'])
    self.assertEqual(options.gdb_debug, True)
    self.assertEqual(options.output_dir, 'my_dir')
    self.check_default_global_options()
    self.check_default_options(options, ['gdb_debug', 'output_dir'])