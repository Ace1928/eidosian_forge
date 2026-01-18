import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_capi_reexport_cincludes(self):
    options, sources = parse_command_line(['--capi-reexport-cincludes', 'source.pyx'])
    self.assertEqual(options.capi_reexport_cincludes, True)
    self.check_default_global_options()
    self.check_default_options(options, ['capi_reexport_cincludes'])