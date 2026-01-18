import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def check_default_options(self, options, white_list=[]):
    default_options = Options.CompilationOptions(Options.default_options)
    no_value = object()
    for name in default_options.__dict__.keys():
        if name not in white_list:
            self.assertEqual(getattr(options, name, no_value), getattr(default_options, name), msg='error in option ' + name)