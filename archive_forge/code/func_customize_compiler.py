import contextlib
import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from distutils import sysconfig
from distutils.ccompiler import get_default_compiler
from distutils.tests import support
from test.support import swap_item, requires_subprocess, is_wasi
from test.support.os_helper import TESTFN
from test.support.warnings_helper import check_warnings
def customize_compiler(self):

    class compiler:
        compiler_type = 'unix'

        def set_executables(self, **kw):
            self.exes = kw
    sysconfig_vars = {'AR': 'sc_ar', 'CC': 'sc_cc', 'CXX': 'sc_cxx', 'ARFLAGS': '--sc-arflags', 'CFLAGS': '--sc-cflags', 'CCSHARED': '--sc-ccshared', 'LDSHARED': 'sc_ldshared', 'SHLIB_SUFFIX': 'sc_shutil_suffix', 'CUSTOMIZED_OSX_COMPILER': 'True'}
    comp = compiler()
    with contextlib.ExitStack() as cm:
        for key, value in sysconfig_vars.items():
            cm.enter_context(swap_item(sysconfig._config_vars, key, value))
        sysconfig.customize_compiler(comp)
    return comp