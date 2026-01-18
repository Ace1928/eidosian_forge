import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_command_case(self):
    cmd = CommandData('cmd')
    cmd.plugin = PluginData('plugger', '1.0')
    bar = OptionData('--bar')
    bar.registry_keys = ['that', 'this']
    bar.error_messages.append('Some error message')
    cmd.options.append(bar)
    cmd.options.append(OptionData('--foo'))
    data = CompletionData()
    data.commands.append(cmd)
    cg = BashCodeGen(data)
    self.assertEqualDiff('\tcmd)\n\t\t# plugin "plugger 1.0"\n\t\t# Some error message\n\t\tcmdOpts=( --bar=that --bar=this --foo )\n\t\tcase $curOpt in\n\t\t\t--bar) optEnums=( that this ) ;;\n\t\tesac\n\t\t;;\n', cg.command_case(cmd))