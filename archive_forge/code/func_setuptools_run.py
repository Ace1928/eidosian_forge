import sys
from distutils.file_util import write_file
def setuptools_run(self):
    """ The setuptools version of the .run() method.

        We must pull in the entire code so we can override the level used in the
        _getframe() call since we wrap this call by one more level.
        """
    from distutils.command.install import install as distutils_install
    if self.old_and_unmanageable or self.single_version_externally_managed:
        return distutils_install.run(self)
    caller = sys._getframe(3)
    caller_module = caller.f_globals.get('__name__', '')
    caller_name = caller.f_code.co_name
    if caller_module != 'distutils.dist' or caller_name != 'run_commands':
        distutils_install.run(self)
    else:
        self.do_egg_install()