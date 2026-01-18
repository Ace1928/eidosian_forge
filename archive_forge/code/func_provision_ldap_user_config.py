from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_ldap_user_config(ldap_user_config, msg):
    """Provision FeatureSpec LdapConfig User from the parsed yaml file.

  Args:
    ldap_user_config: YamlConfigFile, The ldap user data loaded from the yaml
      file given by the user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing the user details of a
    single LDAP auth method for the IdentityServiceFeatureSpec.
  """
    user = msg.IdentityServiceUserConfig()
    if 'baseDn' not in ldap_user_config:
        raise exceptions.Error('LDAP Authentication method must contain user baseDn.')
    user.baseDn = ldap_user_config['baseDn']
    if 'loginAttribute' in ldap_user_config:
        user.loginAttribute = ldap_user_config['loginAttribute']
    if 'idAttribute' in ldap_user_config:
        user.idAttribute = ldap_user_config['idAttribute']
    if 'filter' in ldap_user_config:
        user.filter = ldap_user_config['filter']
    return user