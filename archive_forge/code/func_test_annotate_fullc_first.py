import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_annotate_fullc_first(self):
    options, sources = parse_command_line(['--annotate-fullc', '--annotate', 'file3.pyx'])
    self.assertEqual(Options.annotate, 'default')
    self.check_default_global_options(['annotate'])
    self.check_default_options(options)