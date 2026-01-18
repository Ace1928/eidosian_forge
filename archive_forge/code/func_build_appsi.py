import shutil
import glob
import os
import sys
import tempfile
def build_appsi(args=[]):
    print('\n\n**** Building APPSI ****')
    from setuptools import Distribution
    from pybind11.setup_helpers import build_ext
    import pybind11.setup_helpers
    from pyomo.common.envvar import PYOMO_CONFIG_DIR
    from pyomo.common.fileutils import this_file_dir

    class appsi_build_ext(build_ext):

        def run(self):
            basedir = os.path.abspath(os.path.curdir)
            if self.inplace:
                tmpdir = os.path.join(this_file_dir(), 'cmodel')
            else:
                tmpdir = os.path.abspath(tempfile.mkdtemp())
            print("Building in '%s'" % tmpdir)
            os.chdir(tmpdir)
            try:
                super(appsi_build_ext, self).run()
                if not self.inplace:
                    library = glob.glob('build/*/appsi_cmodel.*')[0]
                    target = os.path.join(PYOMO_CONFIG_DIR, 'lib', 'python%s.%s' % sys.version_info[:2], 'site-packages', '.')
                    if not os.path.exists(target):
                        os.makedirs(target)
                    shutil.copy(library, target)
            finally:
                os.chdir(basedir)
                if not self.inplace:
                    shutil.rmtree(tmpdir, onerror=handleReadonly)
    try:
        original_pybind11_setup_helpers_macos = pybind11.setup_helpers.MACOS
        pybind11.setup_helpers.MACOS = False
        package_config = {'name': 'appsi_cmodel', 'packages': [], 'ext_modules': [get_appsi_extension(False)], 'cmdclass': {'build_ext': appsi_build_ext}}
        dist = Distribution(package_config)
        dist.script_args = ['build_ext'] + args
        dist.parse_command_line()
        dist.run_command('build_ext')
    finally:
        pybind11.setup_helpers.MACOS = original_pybind11_setup_helpers_macos