import logging
import os
import time
import unittest
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_privsep import comm
from oslo_privsep import priv_context
def test_context_with_timeout_pass(self):
    thread_pool_size = self.cfg_fixture.conf.privsep.thread_pool_size
    for _ in range(thread_pool_size + 1):
        res = sleep_with_t_context(0.01)
        self.assertEqual(42, res)