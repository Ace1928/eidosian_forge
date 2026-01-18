import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def assertCompletionOmits(self, *words):
    surplus = set(words) & self.completion_result
    if surplus:
        raise AssertionError('Completion should omit %r but it has %r' % (surplus, self.completion_result))