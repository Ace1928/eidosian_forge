import doctest
import errno
import glob
import logging
import os
import shlex
import sys
import textwrap
from .. import osutils, tests, trace
from ..tests import ui_testing
def run_script(self, script, null_output_matches_anything=False):
    return self.script_runner.run_script(self, script, null_output_matches_anything=null_output_matches_anything)