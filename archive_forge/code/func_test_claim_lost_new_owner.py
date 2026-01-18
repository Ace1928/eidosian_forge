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
def test_claim_lost_new_owner(self):
    job = self._post_claim_job('test')
    f = self._make_dummy_flow(10)
    e = self._make_engine(f)
    change_owner = True
    ran_states = []
    with claims.CheckingClaimListener(e, job, self.board, self.board.name):
        for state in e.run_iter():
            ran_states.append(state)
            if state == states.SCHEDULING and change_owner:
                change_owner = bool(self._change_owner('test-2'))
    self.assertEqual(states.SUSPENDED, e.storage.get_flow_state())
    self.assertEqual(1, ran_states.count(states.ANALYZING))
    self.assertEqual(1, ran_states.count(states.SCHEDULING))
    self.assertEqual(1, ran_states.count(states.WAITING))