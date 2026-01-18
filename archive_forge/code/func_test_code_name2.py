import linecache
import sys
from IPython.core import compilerop
def test_code_name2():
    code = 'x=1'
    name = compilerop.code_name(code, 9)
    assert name.startswith('<ipython-input-9')