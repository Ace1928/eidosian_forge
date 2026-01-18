import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_no_message(self):
    self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
    logger.setLevel(logging.WARNING)
    logger.info('')
    self.assertEqual(self.stream.getvalue(), '')
    logger.warning('')
    ans = 'WARNING:\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.setLevel(logging.DEBUG)
    logger.warning('')
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'WARNING: "[base]%stest_log.py", %d, test_no_message\n\n' % (os.path.sep, lineno)
    self.assertEqual(self.stream.getvalue(), ans)