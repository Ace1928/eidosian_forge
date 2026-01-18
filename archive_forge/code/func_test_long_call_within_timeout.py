import logging
import os
import pipes
import platform
import sys
import tempfile
import time
from unittest import mock
import testtools
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep import priv_context
from oslo_privsep.tests import testctx
def test_long_call_within_timeout(self):
    res = do_some_long(0.001)
    self.assertEqual(42, res)