import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_global_opts(self):
    self.complete(['brz', '-', 'init'], cword=1)
    self.assertCompletionContains('--no-plugins', '--builtin')