import datetime
from testtools import content as ttc
import time
from unittest import mock
import uuid
from oslo_log import log as logging
from oslo_utils import fixture as time_fixture
from oslo_utils import units
from glance.tests import functional
from glance.tests import utils as test_utils
def slow_fake_set_data(data_iter, size=None, backend=None, set_active=True):
    me = str(uuid.uuid4())
    while state['want_run'] == True:
        LOG.info('fake_set_data running %s', me)
        state['running'] = True
        time.sleep(0.1)
    LOG.info('fake_set_data ended %s', me)