import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_commit_dashm(self):
    dc = DataCollector()
    cmd = dc.command('commit')
    self.assertSubset(['-m'], [str(o) for o in cmd.options])