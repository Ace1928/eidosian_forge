import openstack.exceptions as exception
from openstack.identity.v3 import (
from openstack.identity.v3 import access_rule as _access_rule
from openstack.identity.v3 import credential as _credential
from openstack.identity.v3 import domain as _domain
from openstack.identity.v3 import domain_config as _domain_config
from openstack.identity.v3 import endpoint as _endpoint
from openstack.identity.v3 import federation_protocol as _federation_protocol
from openstack.identity.v3 import group as _group
from openstack.identity.v3 import identity_provider as _identity_provider
from openstack.identity.v3 import limit as _limit
from openstack.identity.v3 import mapping as _mapping
from openstack.identity.v3 import policy as _policy
from openstack.identity.v3 import project as _project
from openstack.identity.v3 import region as _region
from openstack.identity.v3 import registered_limit as _registered_limit
from openstack.identity.v3 import role as _role
from openstack.identity.v3 import role_assignment as _role_assignment
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import service as _service
from openstack.identity.v3 import system as _system
from openstack.identity.v3 import trust as _trust
from openstack.identity.v3 import user as _user
from openstack import proxy
from openstack import resource
from openstack import utils
def assign_domain_role_to_group(self, domain, group, role):
    """Assign role to group on a domain

        :param domain: Either the ID of a domain or a
            :class:`~openstack.identity.v3.domain.Domain` instance.
        :param group: Either the ID of a group or a
            :class:`~openstack.identity.v3.group.Group` instance.
        :param role: Either the ID of a role or a
            :class:`~openstack.identity.v3.role.Role` instance.
        :return: ``None``
        """
    domain = self._get_resource(_domain.Domain, domain)
    group = self._get_resource(_group.Group, group)
    role = self._get_resource(_role.Role, role)
    domain.assign_role_to_group(self, group, role)