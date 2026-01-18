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
def test_delete_keypair_notfound(self):
    stack_name = 'stackuser_testdel_kpr_notfound'
    project_id = 'aprojectdel'
    user_id = 'auserdel'
    rsrc = self._user_create(stack_name=stack_name, project_id=project_id, user_id=user_id)
    self.fc.delete_stack_domain_user_keypair.return_value = None
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    rsrc.data_set('credential_id', 'acredential')
    rsrc._delete_keypair()
    rs_data = resource_data_object.ResourceData.get_all(rsrc)
    self.assertEqual({'user_id': user_id}, rs_data)
    self.fc.delete_stack_domain_user_keypair.assert_called_once_with(credential_id='acredential', project_id=project_id, user_id=user_id)