from dill.source import getsource, getname, _wrap, likely_import
from dill.source import getimportable
from dill._dill import IS_PYPY
import sys
def test_builtin():
    assert likely_import(pow) == 'pow\n'
    assert likely_import(100) == '100\n'
    assert likely_import(True) == 'True\n'
    assert likely_import(pow, explicit=True) == 'from builtins import pow\n'
    assert likely_import(100, explicit=True) == '100\n'
    assert likely_import(True, explicit=True) == 'True\n'
    assert likely_import(None) == 'None\n'
    assert likely_import(None, explicit=True) == 'None\n'