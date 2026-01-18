from unittest import mock
import uuid
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_tags(self):
    tags = [{'Key': 'Food', 'Value': 'yum'}]
    metadata = dict(((tm['Key'], tm['Value']) for tm in tags))
    instance = self._setup_test_instance(intags=tags)
    scheduler.TaskRunner(instance.create)()
    self.mock_build_userdata.assert_called_once_with(self.metadata, instance.properties['UserData'], 'ec2-user')
    self.fc.servers.create.assert_called_once_with(image=1, flavor=1, key_name='test', name=utils.PhysName(self.stack_name, instance.name), security_groups=None, userdata=self.server_userdata, scheduler_hints=None, meta=metadata, nics=None, availability_zone=None, block_device_mapping=None)