import copy
import eventlet
import fixtures
import functools
import logging as pylogging
import platform
import sys
import time
from unittest import mock
from oslo_log import formatters
from oslo_log import log as logging
from oslotest import base
import testtools
from oslo_privsep import capabilities
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep.tests import testctx
@testctx.context.entrypoint
def logme(level, msg, exc_info=False):
    LOG.logger.setLevel(logging.DEBUG)
    if exc_info:
        try:
            raise TestException('with arg')
        except TestException:
            LOG.log(level, msg, exc_info=True)
    else:
        LOG.log(level, msg)