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
def role_assignments_filter(self, domain=None, project=None, system=None, group=None, user=None):
    """Retrieve a generator of roles assigned to user/group

        :param domain: Either the ID of a domain or a
            :class:`~openstack.identity.v3.domain.Domain` instance.
        :param project: Either the ID of a project or a
            :class:`~openstack.identity.v3.project.Project`
            instance.
        :param system: Either the system name or a
            :class:`~openstack.identity.v3.system.System`
            instance.
        :param group: Either the ID of a group or a
            :class:`~openstack.identity.v3.group.Group` instance.
        :param user: Either the ID of a user or a
            :class:`~openstack.identity.v3.user.User` instance.
        :return: A generator of role instances.
        :rtype: :class:`~openstack.identity.v3.role.Role`
        """
    if domain and project and system:
        raise exception.InvalidRequest('Only one of domain, project, or system can be specified')
    if domain is None and project is None and (system is None):
        raise exception.InvalidRequest('Either domain, project, or system should be specified')
    if group and user:
        raise exception.InvalidRequest('Only one of group or user can be specified')
    if group is None and user is None:
        raise exception.InvalidRequest('Either group or user should be specified')
    if domain:
        domain_id = resource.Resource._get_id(domain)
        if group:
            group_id = resource.Resource._get_id(group)
            return self._list(_role_domain_group_assignment.RoleDomainGroupAssignment, domain_id=domain_id, group_id=group_id)
        else:
            user_id = resource.Resource._get_id(user)
            return self._list(_role_domain_user_assignment.RoleDomainUserAssignment, domain_id=domain_id, user_id=user_id)
    elif project:
        project_id = resource.Resource._get_id(project)
        if group:
            group_id = resource.Resource._get_id(group)
            return self._list(_role_project_group_assignment.RoleProjectGroupAssignment, project_id=project_id, group_id=group_id)
        else:
            user_id = resource.Resource._get_id(user)
            return self._list(_role_project_user_assignment.RoleProjectUserAssignment, project_id=project_id, user_id=user_id)
    else:
        system_id = resource.Resource._get_id(system)
        if group:
            group_id = resource.Resource._get_id(group)
            return self._list(_role_system_group_assignment.RoleSystemGroupAssignment, system_id=system_id, group_id=group_id)
        else:
            user_id = resource.Resource._get_id(user)
            return self._list(_role_system_user_assignment.RoleSystemUserAssignment, system_id=system_id, user_id=user_id)