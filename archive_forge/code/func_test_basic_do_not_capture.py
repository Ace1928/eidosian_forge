import contextlib
import logging
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import reflection
from zake import fake_client
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.jobs import backends as jobs
from taskflow.listeners import claims
from taskflow.listeners import logging as logging_listeners
from taskflow.listeners import timing
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils
def test_basic_do_not_capture(self):
    flow = lf.Flow('test')
    flow.add(test_utils.ProgressingTask('task1'))
    e = self._make_engine(flow)
    with test_utils.CaptureListener(e, capture_task=False) as capturer:
        e.run()
    expected = ['test.f RUNNING', 'test.f SUCCESS']
    self.assertEqual(expected, capturer.values)