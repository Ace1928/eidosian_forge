from debtcollector import renames
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def add_to_group(self, user, group):
    """Add the specified user as a member of the specified group.

        :param user: the user to be added to the group.
        :type user: str or :class:`keystoneclient.v3.users.User`
        :param group: the group to put the user in.
        :type group: str or :class:`keystoneclient.v3.groups.Group`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        """
    self._require_user_and_group(user, group)
    base_url = '/groups/%s' % base.getid(group)
    return super(UserManager, self).put(base_url=base_url, user_id=base.getid(user))