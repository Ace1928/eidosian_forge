import glob
import os
import subprocess
import sys
import tempfile
import textwrap
from setuptools.command.build_ext import customize_compiler, new_compiler
def compile_test_program(code, extra_preargs=None, extra_postargs=None):
    """Check that some C code can be compiled and run"""
    ccompiler = new_compiler()
    customize_compiler(ccompiler)
    start_dir = os.path.abspath('.')
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)
            with open('test_program.c', 'w') as f:
                f.write(code)
            os.mkdir('objects')
            ccompiler.compile(['test_program.c'], output_dir='objects', extra_postargs=extra_postargs)
            objects = glob.glob(os.path.join('objects', '*' + ccompiler.obj_extension))
            ccompiler.link_executable(objects, 'test_program', extra_preargs=extra_preargs, extra_postargs=extra_postargs)
            if 'PYTHON_CROSSENV' not in os.environ:
                output = subprocess.check_output('./test_program')
                output = output.decode(sys.stdout.encoding or 'utf-8').splitlines()
            else:
                output = []
        except Exception:
            raise
        finally:
            os.chdir(start_dir)
    return output