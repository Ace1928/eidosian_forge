from dill.source import getsource, getname, _wrap, likely_import
from dill.source import getimportable
from dill._dill import IS_PYPY
import sys
def test_itself():
    assert likely_import(likely_import) == 'from dill.source import likely_import\n'