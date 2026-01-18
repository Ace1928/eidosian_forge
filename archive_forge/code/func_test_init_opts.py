import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_init_opts(self):
    self.complete(['brz', 'init', '-'])
    self.assertCompletionContains('-h', '--format=2a')