import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_log_stream(self):
    ls = LogStream(logging.INFO, logging.getLogger('pyomo'))
    LI = LoggingIntercept(level=logging.INFO, formatter=pyomo_formatter)
    with LI as OUT:
        ls.write('hello, world\n')
        self.assertEqual(OUT.getvalue(), 'INFO: hello, world\n')
    with LI as OUT:
        ls.write('line 1\nline 2\n')
        self.assertEqual(OUT.getvalue(), 'INFO: line 1\nINFO: line 2\n')
    with LI as OUT:
        ls.write('line 1\nline 2')
        self.assertEqual(OUT.getvalue(), 'INFO: line 1\n')
    with LI as OUT:
        ls.flush()
        self.assertEqual(OUT.getvalue(), 'INFO: line 2\n')
        ls.flush()
        self.assertEqual(OUT.getvalue(), 'INFO: line 2\n')
    with LI as OUT:
        with LogStream(logging.INFO, logging.getLogger('pyomo')) as ls:
            ls.write('line 1\nline 2')
            self.assertEqual(OUT.getvalue(), 'INFO: line 1\n')
        self.assertEqual(OUT.getvalue(), 'INFO: line 1\nINFO: line 2\n')