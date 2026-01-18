import re
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
def parse_entity_with_domain(self, entity_with_domain, entity_type):
    """Parse keystone entity user/role/project with domain.

        entity_with_domain should be in entity{domain} format.

        Returns a tuple of (entity, domain).
        """
    try:
        match = re.search('\\{(.*?)\\}$', entity_with_domain)
        if match:
            entity = entity_with_domain[:match.start()]
            domain = match.group(1)
            domain = self.get_domain_id(domain)
            return (entity, domain)
        else:
            return (entity_with_domain, None)
    except Exception:
        raise exception.EntityNotFound(entity=entity_type, name=entity_with_domain)