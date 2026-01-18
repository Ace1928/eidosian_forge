import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
@pytest.mark.skipif(not HAVE_COMPILER, reason='Missing compiler')
@pytest.mark.skipif('msvc' in repr(ccompiler.new_compiler()), reason='Fails with MSVC compiler ')
def test_compile2(self):
    tsi = self.c_temp2
    c = customized_ccompiler()
    extra_link_args = tsi.calc_extra_info()['extra_link_args']
    previousDir = os.getcwd()
    try:
        os.chdir(self._dir2)
        c.compile([os.path.basename(self._src2)], output_dir=self._dir2, extra_postargs=extra_link_args)
        assert_(os.path.isfile(self._src2.replace('.c', '.o')))
    finally:
        os.chdir(previousDir)