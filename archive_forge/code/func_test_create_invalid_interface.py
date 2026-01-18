import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import endpoints
def test_create_invalid_interface(self):
    ref = self.new_ref(interface=uuid.uuid4().hex)
    self.assertRaises(exceptions.ValidationError, self.manager.create, **utils.parameterize(ref))