import os
import re
import configparser
import keystone.conf
def symptom_LDAP_user_enabled_emulation_dn_ignored():
    """`[ldap] user_enabled_emulation_dn` is being ignored.

    There is no reason to set this value unless `keystone.conf [ldap]
    user_enabled_emulation` is also enabled.
    """
    return not CONF.ldap.user_enabled_emulation and CONF.ldap.user_enabled_emulation_dn is not None