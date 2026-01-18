import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_cimport_from_pyx(self):
    options, sources = parse_command_line(['--cimport-from-pyx', 'source.pyx'])
    self.assertEqual(Options.cimport_from_pyx, True)
    self.check_default_global_options(['cimport_from_pyx'])
    self.check_default_options(options)