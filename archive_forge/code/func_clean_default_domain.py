from oslo_config import fixture as config_fixture
from keystone.cmd import bootstrap
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests.unit import core
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def clean_default_domain(self):
    PROVIDERS.resource_api.update_domain(CONF.identity.default_domain_id, {'enabled': False})
    PROVIDERS.resource_api.delete_domain(CONF.identity.default_domain_id)