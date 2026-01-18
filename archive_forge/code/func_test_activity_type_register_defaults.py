import boto.swf.layer2
from boto.swf.layer2 import ActivityType, WorkflowType, WorkflowExecution
from tests.unit import unittest
from mock import Mock, ANY
def test_activity_type_register_defaults(self):
    act_type = ActivityType(name='name', domain='test', version='1')
    act_type.register()
    act_type._swf.register_activity_type.assert_called_with('test', 'name', '1', default_task_heartbeat_timeout=ANY, default_task_schedule_to_close_timeout=ANY, default_task_schedule_to_start_timeout=ANY, default_task_start_to_close_timeout=ANY)