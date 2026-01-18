import boto.swf.layer2
from boto.swf.layer2 import Decider, ActivityWorker
from tests.unit import unittest
from mock import Mock
def test_actor_poll_without_tasklist_override(self):
    self.worker.poll()
    self.decider.poll()
    self.worker._swf.poll_for_activity_task.assert_called_with('test', 'test_list')
    self.decider._swf.poll_for_decision_task.assert_called_with('test', 'test_list')