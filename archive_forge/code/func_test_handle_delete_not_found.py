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
def test_handle_delete_not_found(self):
    stack_name = 'stackuser_testdel-notfound'
    project_id = 'aprojectdel2'
    user_id = 'auserdel2'
    rsrc = self._user_create(stack_name=stack_name, project_id=project_id, user_id=user_id)
    self.fc.delete_stack_domain_user.side_effect = kc_exceptions.NotFound()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
    self.fc.delete_stack_domain_user.assert_called_once_with(user_id=user_id, project_id=project_id)