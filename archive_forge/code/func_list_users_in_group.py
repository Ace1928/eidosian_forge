import uuid
import ldap.filter
from oslo_log import log
from oslo_log import versionutils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends.ldap import models
def list_users_in_group(self, group_id, hints):
    users = []
    group_members = self.group.list_group_users(group_id)
    for user_id in self._transform_group_member_ids(group_members):
        try:
            users.append(self.user.get_filtered(user_id))
        except exception.UserNotFound:
            msg = 'Group member `%(user_id)s` for group `%(group_id)s` not found in the directory. The user should be removed from the group. The user will be ignored.'
            LOG.debug(msg, dict(user_id=user_id, group_id=group_id))
    return users