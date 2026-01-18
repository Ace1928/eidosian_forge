import os
import re
import configparser
import keystone.conf
def symptom_LDAP_group_members_are_ids_disabled():
    """`[ldap] group_members_are_ids` is not enabled.

    Because you've set `keystone.conf [ldap] group_objectclass = posixGroup`,
    we would have also expected you to enable set `keystone.conf [ldap]
    group_members_are_ids` because we suspect you're using Open Directory,
    which would contain user ID's in a `posixGroup` rather than LDAP DNs, as
    other object classes typically would.
    """
    return CONF.ldap.group_objectclass == 'posixGroup' and (not CONF.ldap.group_members_are_ids)