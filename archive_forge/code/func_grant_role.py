from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def grant_role(self, name_or_id, user=None, group=None, project=None, domain=None, system=None, wait=False, timeout=60):
    """Grant a role to a user.

        :param string name_or_id: Name or unique ID of the role.
        :param string user: The name or id of the user.
        :param string group: The name or id of the group. (v3)
        :param string project: The name or id of the project.
        :param string domain: The id of the domain. (v3)
        :param bool system: The name of the system. (v3)
        :param bool wait: Wait for role to be granted
        :param int timeout: Timeout to wait for role to be granted

         NOTE: domain is a required argument when the grant is on a project,
         user or group specified by name. In that situation, they are all
         considered to be in that domain. If different domains are in use in
         the same role grant, it is required to specify those by ID.

         NOTE: for wait and timeout, sometimes granting roles is not
         instantaneous.

        NOTE: precedence is given first to project, then domain, then system

        :returns: True if the role is assigned, otherwise False
        :raises: :class:`~openstack.exceptions.SDKException` if the role cannot
            be granted
        """
    data = self._get_grant_revoke_params(name_or_id, user=user, group=group, project=project, domain=domain, system=system)
    user = data.get('user')
    group = data.get('group')
    project = data.get('project')
    domain = data.get('domain')
    role = data.get('role')
    if project:
        if user:
            has_role = self.identity.validate_user_has_project_role(project, user, role)
            if has_role:
                self.log.debug('Assignment already exists')
                return False
            self.identity.assign_project_role_to_user(project, user, role)
        else:
            has_role = self.identity.validate_group_has_project_role(project, group, role)
            if has_role:
                self.log.debug('Assignment already exists')
                return False
            self.identity.assign_project_role_to_group(project, group, role)
    elif domain:
        if user:
            has_role = self.identity.validate_user_has_domain_role(domain, user, role)
            if has_role:
                self.log.debug('Assignment already exists')
                return False
            self.identity.assign_domain_role_to_user(domain, user, role)
        else:
            has_role = self.identity.validate_group_has_domain_role(domain, group, role)
            if has_role:
                self.log.debug('Assignment already exists')
                return False
            self.identity.assign_domain_role_to_group(domain, group, role)
    elif user:
        has_role = self.identity.validate_user_has_system_role(user, role, system)
        if has_role:
            self.log.debug('Assignment already exists')
            return False
        self.identity.assign_system_role_to_user(user, role, system)
    else:
        has_role = self.identity.validate_group_has_system_role(group, role, system)
        if has_role:
            self.log.debug('Assignment already exists')
            return False
        self.identity.assign_system_role_to_group(group, role, system)
    return True