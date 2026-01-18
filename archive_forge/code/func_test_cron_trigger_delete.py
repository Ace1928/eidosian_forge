from openstack.tests.unit import test_proxy_base
from openstack.workflow.v2 import _proxy
from openstack.workflow.v2 import cron_trigger
from openstack.workflow.v2 import execution
from openstack.workflow.v2 import workflow
def test_cron_trigger_delete(self):
    self.verify_delete(self.proxy.delete_cron_trigger, cron_trigger.CronTrigger, True)