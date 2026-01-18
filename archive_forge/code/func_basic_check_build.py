import glob
import os
import subprocess
import sys
import tempfile
import textwrap
from setuptools.command.build_ext import customize_compiler, new_compiler
def basic_check_build():
    """Check basic compilation and linking of C code"""
    if 'PYODIDE_PACKAGE_ABI' in os.environ:
        return
    code = textwrap.dedent('        #include <stdio.h>\n        int main(void) {\n        return 0;\n        }\n        ')
    compile_test_program(code)