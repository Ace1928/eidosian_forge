import logging
import os
from inspect import currentframe, getframeinfo
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import (
def test_long_messages(self):
    self.handler.setFormatter(LegacyPyomoFormatter(base=os.path.dirname(__file__), verbosity=lambda: logger.isEnabledFor(logging.DEBUG)))
    msg = 'This is a long message\n\nWith some kind of internal formatting\n    - including a bulleted list\n    - list 2  '
    logger.setLevel(logging.WARNING)
    logger.warning(msg)
    ans = 'WARNING: This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.setLevel(logging.DEBUG)
    logger.info(msg)
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'INFO: "[base]%stest_log.py", %d, test_long_messages\n    This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n' % (os.path.sep, lineno)
    self.assertEqual(self.stream.getvalue(), ans)
    msg += '\n'
    logger.setLevel(logging.WARNING)
    logger.warning(msg)
    ans += 'WARNING: This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.setLevel(logging.DEBUG)
    logger.info(msg)
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'INFO: "[base]%stest_log.py", %d, test_long_messages\n    This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n' % (os.path.sep, lineno)
    self.assertEqual(self.stream.getvalue(), ans)
    msg = '\n' + msg + '\n\n'
    logger.setLevel(logging.WARNING)
    logger.warning(msg)
    ans += 'WARNING: This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n'
    self.assertEqual(self.stream.getvalue(), ans)
    logger.setLevel(logging.DEBUG)
    logger.info(msg)
    lineno = getframeinfo(currentframe()).lineno - 1
    ans += 'INFO: "[base]%stest_log.py", %d, test_long_messages\n    This is a long message\n\n    With some kind of internal formatting\n        - including a bulleted list\n        - list 2\n' % (os.path.sep, lineno)
    self.assertEqual(self.stream.getvalue(), ans)