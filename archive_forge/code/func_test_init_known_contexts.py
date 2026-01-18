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
def test_init_known_contexts(self):
    self.assertEqual(testctx.context.helper_command('/sock')[:2], ['sudo', 'privsep-helper'])
    priv_context.init(root_helper=['sudo', 'rootwrap'])
    self.assertEqual(testctx.context.helper_command('/sock')[:3], ['sudo', 'rootwrap', 'privsep-helper'])