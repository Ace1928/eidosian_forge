import abc
import keystone.conf
from keystone import exception
@property
def multiple_domains_supported(self):
    return self.is_domain_aware() or CONF.identity.domain_specific_drivers_enabled