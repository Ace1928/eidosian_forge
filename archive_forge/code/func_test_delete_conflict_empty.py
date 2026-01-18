from unittest import mock
from oslo_config import cfg
import swiftclient.client as sc
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.aws.s3 import s3
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_delete_conflict_empty(self):
    cfg.CONF.set_override('action_retry_limit', 0)
    t = template_format.parse(swift_template)
    stack = utils.parse_stack(t)
    container_name = utils.PhysName(stack.name, 'test_resource')
    self.mock_con.put_container.return_value = None
    self.mock_con.delete_container.side_effect = sc.ClientException('Conflict', http_status=409)
    self.mock_con.get_container.return_value = ({'name': container_name}, [])
    rsrc = self.create_resource(t, stack, 'S3Bucket')
    deleter = scheduler.TaskRunner(rsrc.delete)
    ex = self.assertRaises(exception.ResourceFailure, deleter)
    self.assertIn('Conflict', str(ex))
    self.mock_con.put_container.assert_called_once_with(container_name, {'X-Container-Write': 'test_tenant:test_username', 'X-Container-Read': 'test_tenant:test_username'})
    self.mock_con.delete_container.assert_called_once_with(container_name)
    self.mock_con.get_container.assert_called_once_with(container_name)