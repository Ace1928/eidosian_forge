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
def check_user_in_group(self, user_id, group_id):
    self.get_user(user_id)
    member_list = self.group.list_group_users(group_id)
    for group_member_id in self._transform_group_member_ids(member_list):
        if group_member_id == user_id:
            break
    else:
        raise exception.NotFound(_("User '%(user_id)s' not found in group '%(group_id)s'") % {'user_id': user_id, 'group_id': group_id})