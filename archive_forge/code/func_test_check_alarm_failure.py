import copy
import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.clients.os import octavia
from heat.engine import resource
from heat.engine.resources.openstack.aodh import alarm
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_check_alarm_failure(self):
    res = self._prepare_resource()
    res.client().alarm.get.side_effect = Exception('Boom')
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.check))
    self.assertEqual((res.CHECK, res.FAILED), res.state)
    self.assertIn('Boom', res.status_reason)