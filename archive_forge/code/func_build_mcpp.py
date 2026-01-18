import os
import shutil
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_dir
from pyomo.common.download import FileDownloader
def build_mcpp():
    from setuptools import Distribution
    from setuptools.command.build_ext import build_ext

    class _BuildWithoutPlatformInfo(build_ext, object):

        def get_ext_filename(self, ext_name):
            filename = super(_BuildWithoutPlatformInfo, self).get_ext_filename(ext_name).split('.')
            filename = '.'.join([filename[0], filename[-1]])
            return filename
    print('\n**** Building MCPP library ****')
    package_config = _generate_configuration()
    package_config['cmdclass'] = {'build_ext': _BuildWithoutPlatformInfo}
    dist = Distribution(package_config)
    install_dir = os.path.join(envvar.PYOMO_CONFIG_DIR, 'lib')
    dist.get_command_obj('install_lib').install_dir = install_dir
    try:
        basedir = os.path.abspath(os.path.curdir)
        tmpdir = os.path.abspath(tempfile.mkdtemp())
        print('   tmpdir = %s' % (tmpdir,))
        os.chdir(tmpdir)
        dist.run_command('install_lib')
        print('Installed mcppInterface to %s' % (install_dir,))
    finally:
        os.chdir(basedir)
        shutil.rmtree(tmpdir)