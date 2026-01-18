import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
def test_ipy_script_file_attribute(self):
    """Test that `__file__` is set when running `ipython file.ipy`"""
    src = 'print(__file__)\n'
    self.mktmp(src, ext='.ipy')
    err = None
    tt.ipexec_validate(self.fname, self.fname, err)