import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_default_domain_id_with_v3(self, **kwargs):
    self.stub_discovery()
    project_name = uuid.uuid4().hex
    default_domain_id = kwargs.setdefault('default_domain_id', uuid.uuid4().hex)
    p = self.assertCreateV3(project_name=project_name, **kwargs)
    self.assertEqual(default_domain_id, p._plugin.project_domain_id)
    self.assertEqual(project_name, p._plugin.project_name)
    return p