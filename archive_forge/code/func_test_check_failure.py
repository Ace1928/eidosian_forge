from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_check_failure(self):
    res = self._prepare_resource()
    res.client().alarm.get.side_effect = Exception('Boom')
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.check))
    self.assertEqual((res.CHECK, res.FAILED), res.state)
    self.assertIn('Boom', res.status_reason)