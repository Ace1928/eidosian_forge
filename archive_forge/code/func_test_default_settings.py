import os
import unittest
from distutils.command.install_scripts import install_scripts
from distutils.core import Distribution
from distutils.tests import support
def test_default_settings(self):
    dist = Distribution()
    dist.command_obj['build'] = support.DummyCommand(build_scripts='/foo/bar')
    dist.command_obj['install'] = support.DummyCommand(install_scripts='/splat/funk', force=1, skip_build=1)
    cmd = install_scripts(dist)
    self.assertFalse(cmd.force)
    self.assertFalse(cmd.skip_build)
    self.assertIsNone(cmd.build_dir)
    self.assertIsNone(cmd.install_dir)
    cmd.finalize_options()
    self.assertTrue(cmd.force)
    self.assertTrue(cmd.skip_build)
    self.assertEqual(cmd.build_dir, '/foo/bar')
    self.assertEqual(cmd.install_dir, '/splat/funk')