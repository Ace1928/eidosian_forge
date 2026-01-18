import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_alternate_base(self):
    self.handler.setFormatter(LegacyPyomoFormatter(base='log_config'))
    logger.setLevel(logging.WARNING)
    logger.info('(info)')
    self.assertEqual(self.stream.getvalue(), '')
    logger.warning('(warn)')
    lineno = getframeinfo(currentframe()).lineno - 1
    ans = 'WARNING: "%s", %d, test_alternate_base\n    (warn)\n' % (filename, lineno)
    self.assertEqual(self.stream.getvalue(), ans)