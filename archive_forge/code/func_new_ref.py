import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import registered_limits
def new_ref(self, **kwargs):
    ref = {'id': uuid.uuid4().hex, 'service_id': uuid.uuid4().hex, 'resource_name': uuid.uuid4().hex, 'default_limit': 10, 'description': uuid.uuid4().hex}
    ref.update(kwargs)
    return ref