import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def ensure_directories(self, env_dir):
    """
        Create the directories for the environment.

        Returns a context object which holds paths in the environment,
        for use by subsequent logic.
        """

    def create_if_needed(d):
        if not os.path.exists(d):
            os.makedirs(d)
        elif os.path.islink(d) or os.path.isfile(d):
            raise ValueError('Unable to create directory %r' % d)
    if os.pathsep in os.fspath(env_dir):
        raise ValueError(f'Refusing to create a venv in {env_dir} because it contains the PATH separator {os.pathsep}.')
    if os.path.exists(env_dir) and self.clear:
        self.clear_directory(env_dir)
    context = types.SimpleNamespace()
    context.env_dir = env_dir
    context.env_name = os.path.split(env_dir)[1]
    prompt = self.prompt if self.prompt is not None else context.env_name
    context.prompt = '(%s) ' % prompt
    create_if_needed(env_dir)
    executable = sys._base_executable
    if not executable:
        raise ValueError('Unable to determine path to the running Python interpreter. Provide an explicit path or check that your PATH environment variable is correctly set.')
    dirname, exename = os.path.split(os.path.abspath(executable))
    context.executable = executable
    context.python_dir = dirname
    context.python_exe = exename
    binpath = self._venv_path(env_dir, 'scripts')
    incpath = self._venv_path(env_dir, 'include')
    libpath = self._venv_path(env_dir, 'purelib')
    context.inc_path = incpath
    create_if_needed(incpath)
    create_if_needed(libpath)
    if sys.maxsize > 2 ** 32 and os.name == 'posix' and (sys.platform != 'darwin'):
        link_path = os.path.join(env_dir, 'lib64')
        if not os.path.exists(link_path):
            os.symlink('lib', link_path)
    context.bin_path = binpath
    context.bin_name = os.path.relpath(binpath, env_dir)
    context.env_exe = os.path.join(binpath, exename)
    create_if_needed(binpath)
    context.env_exec_cmd = context.env_exe
    if sys.platform == 'win32':
        real_env_exe = os.path.realpath(context.env_exe)
        if os.path.normcase(real_env_exe) != os.path.normcase(context.env_exe):
            logger.warning('Actual environment location may have moved due to redirects, links or junctions.\n  Requested location: "%s"\n  Actual location:    "%s"', context.env_exe, real_env_exe)
            context.env_exec_cmd = real_env_exe
    return context