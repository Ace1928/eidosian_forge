import os
import unittest
from distutils.command.build_scripts import build_scripts
from distutils.core import Distribution
from distutils import sysconfig
from distutils.tests import support
def get_build_scripts_cmd(self, target, scripts):
    import sys
    dist = Distribution()
    dist.scripts = scripts
    dist.command_obj['build'] = support.DummyCommand(build_scripts=target, force=1, executable=sys.executable)
    return build_scripts(dist)