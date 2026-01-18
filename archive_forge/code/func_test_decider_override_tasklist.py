import boto.swf.layer2
from boto.swf.layer2 import Decider, ActivityWorker
from tests.unit import unittest
from mock import Mock
def test_decider_override_tasklist(self):
    self.decider.poll(task_list='some_other_tasklist')
    self.decider._swf.poll_for_decision_task.assert_called_with('test', 'some_other_tasklist')