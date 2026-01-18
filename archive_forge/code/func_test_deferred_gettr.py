import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
def test_deferred_gettr(self):
    api_name = '%s_api' % uuid.uuid4().hex

    class TestClass(object):
        descriptor = provider_api.ProviderAPIs.deferred_provider_lookup(api=api_name, method='do_something')
    test_instance = TestClass()
    self.assertRaises(AttributeError, getattr, test_instance, 'descriptor')
    self._create_manager_instance(provides_api=api_name)
    self.assertEqual(api_name, test_instance.descriptor())