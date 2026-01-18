import errno
import os
import shutil
import stat
import sys
import tempfile
import pyomo.common.envvar as envvar
from pyomo.common.fileutils import this_file_dir, find_executable
def build_cmake_project(targets, package_name=None, description=None, user_args=[], parallel=None):
    from setuptools import Extension, Distribution
    from setuptools.command.build_ext import build_ext

    class _CMakeBuild(build_ext, object):

        def run(self):
            for cmake_ext in self.extensions:
                self._cmake_build_target(cmake_ext)

        def _cmake_build_target(self, cmake_ext):
            cmake_config = 'Debug' if self.debug else 'Release'
            cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + envvar.PYOMO_CONFIG_DIR] + cmake_ext.user_args
            try:
                sys.stderr.flush()
                sys.stdout.flush()
                old_stderr = os.dup(sys.stderr.fileno())
                os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
                old_environ = dict(os.environ)
                if cmake_ext.parallel:
                    os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = str(cmake_ext.parallel)
                cmake = find_executable('cmake')
                if cmake is None:
                    raise IOError('cmake not found in the system PATH')
                self.spawn([cmake, cmake_ext.target_dir] + cmake_args)
                if not self.dry_run:
                    self.spawn([cmake, '--build', '.', '--target', 'install', '--config', cmake_config])
            finally:
                sys.stderr.flush()
                sys.stdout.flush()
                os.dup2(old_stderr, sys.stderr.fileno())
                os.environ = old_environ

    class CMakeExtension(Extension, object):

        def __init__(self, target_dir, user_args, parallel):
            super(CMakeExtension, self).__init__(self.__class__.__qualname__, sources=[])
            self.target_dir = target_dir
            self.user_args = user_args
            self.parallel = parallel
    if package_name is None:
        package_name = 'build_cmake'
    if description is None:
        description = package_name
    caller_dir = this_file_dir(2)
    ext_modules = [CMakeExtension(os.path.join(caller_dir, target), user_args, parallel) for target in targets]
    sys.stdout.write(f'\n**** Building {description} ****\n')
    package_config = {'name': package_name, 'packages': [], 'ext_modules': ext_modules, 'cmdclass': {'build_ext': _CMakeBuild}}
    dist = Distribution(package_config)
    basedir = os.path.abspath(os.path.curdir)
    try:
        tmpdir = os.path.abspath(tempfile.mkdtemp())
        os.chdir(tmpdir)
        dist.run_command('build_ext')
        install_dir = envvar.PYOMO_CONFIG_DIR
    finally:
        os.chdir(basedir)
        shutil.rmtree(tmpdir, onerror=handleReadonly)
    sys.stdout.write(f'Installed {description} to {install_dir}\n')