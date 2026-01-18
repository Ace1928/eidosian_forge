from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavor_access
def get_expected(flavor_access):
    return '<FlavorAccess flavor id: %s, tenant id: %s>' % (flavor_access.flavor_id, flavor_access.tenant_id)