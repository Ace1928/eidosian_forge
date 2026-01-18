from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def connect_to_ldap(module, server_uri, bind_dn=None, bind_pw=None, insecure=True, ca_file=None):
    if insecure:
        ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
    elif ca_file:
        ldap.set_option(ldap.OPT_X_TLS_CACERTFILE, ca_file)
    try:
        connection = ldap.initialize(server_uri)
        connection.set_option(ldap.OPT_REFERRALS, 0)
        connection.simple_bind_s(bind_dn, bind_pw)
        return connection
    except ldap.LDAPError as e:
        module.fail_json(msg="Cannot bind to the LDAP server '{0}' due to: {1}".format(server_uri, e))