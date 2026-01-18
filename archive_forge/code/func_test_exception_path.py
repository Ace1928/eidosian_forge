import os
import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
def test_exception_path(self):
    """Test exception path in exception_validate.
        """
    self.mktmp("import sys\nprint('A')\nprint('B')\nprint('C', file=sys.stderr)\nprint('D', file=sys.stderr)\n")
    out = 'A\nB'
    tt.ipexec_validate(self.fname, expected_out=out, expected_err='C\nD')