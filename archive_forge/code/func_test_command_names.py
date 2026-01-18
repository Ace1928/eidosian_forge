import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_command_names(self):
    data = CompletionData()
    bar = CommandData('bar')
    bar.aliases.append('baz')
    data.commands.append(bar)
    data.commands.append(CommandData('foo'))
    cg = BashCodeGen(data)
    self.assertEqual('bar baz foo', cg.command_names())