import logging
import os
import time
import unittest
from oslo_config import fixture as config_fixture
from oslotest import base
from oslo_privsep import comm
from oslo_privsep import priv_context
@test_context.entrypoint_with_timeout(0.03)
def sleep_with_timeout(long_timeout=0.04):
    time.sleep(long_timeout)
    return 42