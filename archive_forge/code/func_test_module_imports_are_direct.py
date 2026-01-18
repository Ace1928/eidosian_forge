import subprocess
import sys
from os.path import abspath, dirname, join, normpath
import inspect
from sympy.testing.pytest import XFAIL
@XFAIL
def test_module_imports_are_direct():
    my_filename = abspath(inspect.getfile(inspect.currentframe()))
    my_dirname = dirname(my_filename)
    diagnose_imports_filename = join(my_dirname, 'diagnose_imports.py')
    diagnose_imports_filename = normpath(diagnose_imports_filename)
    process = subprocess.Popen([sys.executable, normpath(diagnose_imports_filename), '--problems', '--by-importer'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=-1)
    output, _ = process.communicate()
    assert output == '', 'There are import problems:\n' + output.decode()