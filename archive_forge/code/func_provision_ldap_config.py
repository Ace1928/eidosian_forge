from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_ldap_config(auth_method, msg):
    """Provision FeatureSpec LdapConfig from the parsed yaml file.

  Args:
    auth_method: YamlConfigFile, The data loaded from the yaml file given by the
      user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing a single
    LDAP auth method for the IdentityServiceFeatureSpec.
  """
    if 'name' not in auth_method:
        raise exceptions.Error('LDAP Authentication method must contain name.')
    auth_method_proto = msg.IdentityServiceAuthMethod()
    auth_method_proto.name = auth_method['name']
    if 'proxy' in auth_method:
        auth_method_proto.proxy = auth_method['proxy']
    ldap_config = auth_method['ldap']
    if 'server' not in ldap_config or 'user' not in ldap_config or 'serviceAccount' not in ldap_config:
        err_msg = 'Authentication method [{}] must contain server, user and serviceAccount details.'.format(auth_method['name'])
        raise exceptions.Error(err_msg)
    auth_method_proto.ldapConfig = msg.IdentityServiceLdapConfig()
    auth_method_proto.ldapConfig.server = provision_ldap_server_config(ldap_config['server'], msg)
    auth_method_proto.ldapConfig.serviceAccount = provision_ldap_service_account_config(ldap_config['serviceAccount'], msg)
    auth_method_proto.ldapConfig.user = provision_ldap_user_config(ldap_config['user'], msg)
    if 'group' in ldap_config:
        auth_method_proto.ldapConfig.group = provision_ldap_group_config(ldap_config['group'], msg)
    return auth_method_proto