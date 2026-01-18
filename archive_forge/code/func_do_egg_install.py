from distutils.errors import DistutilsArgError
import inspect
import glob
import platform
import distutils.command.install as orig
import setuptools
from ..warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
def do_egg_install(self):
    easy_install = self.distribution.get_command_class('easy_install')
    cmd = easy_install(self.distribution, args='x', root=self.root, record=self.record)
    cmd.ensure_finalized()
    cmd.always_copy_from = '.'
    cmd.package_index.scan(glob.glob('*.egg'))
    self.run_command('bdist_egg')
    args = [self.distribution.get_command_obj('bdist_egg').egg_output]
    if setuptools.bootstrap_install_from:
        args.insert(0, setuptools.bootstrap_install_from)
    cmd.args = args
    cmd.run(show_deprecation=False)
    setuptools.bootstrap_install_from = None