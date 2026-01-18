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
def test_mem_alarm_suspend_resume(self):
    """Tests suspending and resuming of the alarm.

        Make sure that the Alarm resource gets disabled on suspend
        and re-enabled on resume.
        """
    test_stack = self.create_stack()
    update_mock = self.patchobject(self.fa.alarm, 'update')
    al_suspend = {'enabled': False}
    al_resume = {'enabled': True}
    test_stack.create()
    rsrc = test_stack['MEMAlarmHigh']
    scheduler.TaskRunner(rsrc.suspend)()
    self.assertEqual((rsrc.SUSPEND, rsrc.COMPLETE), rsrc.state)
    scheduler.TaskRunner(rsrc.resume)()
    self.assertEqual((rsrc.RESUME, rsrc.COMPLETE), rsrc.state)
    update_mock.assert_has_calls((mock.call('foo', al_suspend), mock.call('foo', al_resume)))