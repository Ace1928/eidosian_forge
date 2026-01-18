import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def new_test(arch, cc):
    if is_standalone:
        return textwrap.dedent('    class TestCCompilerOpt_{class_name}(_Test_CCompilerOpt, unittest.TestCase):\n        arch = \'{arch}\'\n        cc   = \'{cc}\'\n        def __init__(self, methodName="runTest"):\n            unittest.TestCase.__init__(self, methodName)\n            self.setup_class()\n    ').format(class_name=arch + '_' + cc, arch=arch, cc=cc)
    return textwrap.dedent("    class TestCCompilerOpt_{class_name}(_Test_CCompilerOpt):\n        arch = '{arch}'\n        cc   = '{cc}'\n    ").format(class_name=arch + '_' + cc, arch=arch, cc=cc)