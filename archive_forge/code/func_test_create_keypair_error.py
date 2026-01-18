from unittest import mock
from keystoneauth1 import exceptions as kc_exceptions
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
def test_create_keypair_error(self):
    stack_name = 'stackuser_test_cr_keypair-err'
    project_id = 'aprojectdel'
    user_id = 'auserdel'
    rsrc = self._user_create(stack_name=stack_name, project_id=project_id, user_id=user_id)
    self.fc.create_stack_domain_user_keypair.return_value = None
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertRaises(exception.Error, rsrc._create_keypair)
    self.fc.create_stack_domain_user_keypair.assert_called_once_with(project_id=project_id, user_id=user_id)