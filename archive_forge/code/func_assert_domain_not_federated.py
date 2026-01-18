from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def assert_domain_not_federated(self, domain_id, domain):
    """Assert the Domain's name and id do not match the reserved keyword.

        Note that the reserved keyword is defined in the configuration file,
        by default, it is 'Federated', it is also case insensitive.
        If config's option is empty the default hardcoded value 'Federated'
        will be used.

        :raise AssertionError: if domain named match the value in the config.

        """
    federated_domain = CONF.federation.federated_domain_name.lower()
    if domain.get('name') and domain['name'].lower() == federated_domain:
        raise AssertionError(_('Domain cannot be named %s') % domain['name'])
    if domain_id.lower() == federated_domain:
        raise AssertionError(_('Domain cannot have ID %s') % domain_id)