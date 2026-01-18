import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_no_annotate(self):
    options, sources = parse_command_line(['--embed=huhu', 'source.pyx'])
    self.assertFalse(Options.annotate)