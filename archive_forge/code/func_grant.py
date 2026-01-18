from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def grant(self, role, user=None, group=None, system=None, domain=None, project=None, os_inherit_extension_inherited=False, **kwargs):
    """Grant a role to a user or group on a domain or project.

        :param role: the role to be granted on the server.
        :type role: str or :class:`keystoneclient.v3.roles.Role`
        :param user: the specified user to have the role granted on a resource.
                     Domain or project must be specified. User and group are
                     mutually exclusive.
        :type user: str or :class:`keystoneclient.v3.users.User`
        :param group: the specified group to have the role granted on a
                      resource. Domain or project must be specified.
                      User and group are mutually exclusive.
        :type group: str or :class:`keystoneclient.v3.groups.Group`
        :param system: system information to grant the role on. Project,
                       domain, and system are mutually exclusive.
        :type system: str
        :param domain: the domain in which the role will be granted. Either
                       user or group must be specified. Project, domain, and
                       system are mutually exclusive.
        :type domain: str or :class:`keystoneclient.v3.domains.Domain`
        :param project: the project in which the role will be granted. Either
                       user or group must be specified. Project, domain, and
                       system are mutually exclusive.
        :type project: str or :class:`keystoneclient.v3.projects.Project`
        :param bool os_inherit_extension_inherited: OS-INHERIT will be used.
                                                    It provides the ability for
                                                    projects to inherit role
                                                    assignments from their
                                                    domains or from parent
                                                    projects in the hierarchy.
        :param kwargs: any other attribute provided will be passed to server.

        :returns: the granted role returned from server.
        :rtype: :class:`keystoneclient.v3.roles.Role`

        """
    self._enforce_mutually_exclusive_group(system, domain, project)
    self._require_user_xor_group(user, group)
    if os_inherit_extension_inherited:
        kwargs['tail'] = '/inherited_to_projects'
    base_url = self._role_grants_base_url(user, group, system, domain, project, os_inherit_extension_inherited)
    return super(RoleManager, self).put(base_url=base_url, role_id=base.getid(role), **kwargs)