from __future__ import absolute_import
from boto.mturk.test.support import unittest
def skip_unless_has_selenium():
    res = has_selenium()
    if not res:
        return unittest.skip(res.message)
    return identity