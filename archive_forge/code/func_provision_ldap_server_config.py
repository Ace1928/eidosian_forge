from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_ldap_server_config(ldap_server_config, msg):
    """Provision FeatureSpec LdapConfig Server from the parsed yaml file.

  Args:
    ldap_server_config: YamlConfigFile, The ldap server data loaded from the
      yaml file given by the user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing server details of a
    single LDAP auth method for the IdentityServiceFeatureSpec.
  """
    server = msg.IdentityServiceServerConfig()
    if 'host' not in ldap_server_config:
        raise exceptions.Error('LDAP Authentication method must contain server host.')
    server.host = ldap_server_config['host']
    if 'connectionType' in ldap_server_config:
        server.connectionType = ldap_server_config['connectionType']
    if 'certificateAuthorityData' in ldap_server_config:
        server.certificateAuthorityData = bytes(ldap_server_config['certificateAuthorityData'], 'utf-8')
    return server