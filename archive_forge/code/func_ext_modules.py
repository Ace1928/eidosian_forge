from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
@property
def ext_modules(self):
    """
        Check if all modules specified in the value for 'ext_modules' can be build.
        That value (if not None) is a list of dicts with 'name', 'src', 'lib'
        Optional 'test' can be used to make sure trying to compile will work on the host

        creates and return the external modules as Extensions, unless that
        is not necessary at all for the action (like --version)

        test existence of compiler by using export CC=nonexistent; export CXX=nonexistent
        """
    if hasattr(self, '_ext_modules'):
        return self._ext_modules
    if '--version' in sys.argv:
        return None
    if platform.python_implementation() == 'Jython':
        return None
    try:
        plat = sys.argv.index('--plat-name')
        if 'win' in sys.argv[plat + 1]:
            return None
    except ValueError:
        pass
    self._ext_modules = []
    no_test_compile = False
    if '--restructuredtext' in sys.argv:
        no_test_compile = True
    elif 'sdist' in sys.argv:
        no_test_compile = True
    if no_test_compile:
        for target in self._pkg_data.get('ext_modules', []):
            ext = Extension(self.pn(target['name']), sources=[self.pn(x) for x in target['src']], libraries=[self.pn(x) for x in target.get('lib')])
            self._ext_modules.append(ext)
        return self._ext_modules
    print('sys.argv', sys.argv)
    import tempfile
    import shutil
    from textwrap import dedent
    import distutils.sysconfig
    import distutils.ccompiler
    from distutils.errors import CompileError, LinkError
    for target in self._pkg_data.get('ext_modules', []):
        ext = Extension(self.pn(target['name']), sources=[self.pn(x) for x in target['src']], libraries=[self.pn(x) for x in target.get('lib')])
        if 'test' not in target:
            self._ext_modules.append(ext)
            continue
        if sys.version_info[:2] == (3, 4) and platform.system() == 'Windows':
            if 'FORCE_C_BUILD_TEST' not in os.environ:
                self._ext_modules.append(ext)
                continue
        c_code = dedent(target['test'])
        try:
            tmp_dir = tempfile.mkdtemp(prefix='tmp_ruamel_')
            bin_file_name = 'test' + self.pn(target['name'])
            print('test compiling', bin_file_name)
            file_name = os.path.join(tmp_dir, bin_file_name + '.c')
            with open(file_name, 'w') as fp:
                fp.write(c_code)
            compiler = distutils.ccompiler.new_compiler()
            assert isinstance(compiler, distutils.ccompiler.CCompiler)
            distutils.sysconfig.customize_compiler(compiler)
            compiler.add_include_dir(os.getcwd())
            if sys.version_info < (3,):
                tmp_dir = tmp_dir.encode('utf-8')
            compile_out_dir = tmp_dir
            try:
                compiler.link_executable(compiler.compile([file_name], output_dir=compile_out_dir), bin_file_name, output_dir=tmp_dir, libraries=ext.libraries)
            except CompileError:
                debug('compile error:', file_name)
                print('compile error:', file_name)
                continue
            except LinkError:
                debug('link error', file_name)
                print('link error', file_name)
                continue
            self._ext_modules.append(ext)
        except Exception as e:
            debug('Exception:', e)
            print('Exception:', e)
            if sys.version_info[:2] == (3, 4) and platform.system() == 'Windows':
                traceback.print_exc()
        finally:
            shutil.rmtree(tmp_dir)
    return self._ext_modules