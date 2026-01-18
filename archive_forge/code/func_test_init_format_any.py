import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_init_format_any(self):
    self.complete(['brz', 'init', '--format', '=', 'directory'], cword=3)
    self.assertCompletionContains('1.9', '2a')