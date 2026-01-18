from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_ldap_service_account_config(ldap_service_account_config, msg):
    """Provision FeatureSpec LdapConfig ServiceAccount from the parsed yaml file.

  Args:
    ldap_service_account_config: YamlConfigFile, The ldap service account data
      loaded from the yaml file given by the user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing the service account
     details of a single LDAP auth method for the IdentityServiceFeatureSpec.
  """
    if ldap_service_account_config is None:
        raise exceptions.Error('LDAP Authentication method must contain Service Account details.')
    service_account = msg.IdentityServiceServiceAccountConfig()
    if 'simpleBindCredentials' in ldap_service_account_config:
        service_account.simpleBindCredentials = msg.IdentityServiceSimpleBindCredentials()
        ldap_simple_bind_credentials = ldap_service_account_config['simpleBindCredentials']
        if not ldap_simple_bind_credentials['dn'] or not ldap_simple_bind_credentials['password']:
            raise exceptions.Error('LDAP Authentication method must contain non-empty Service Account credentials.')
        service_account.simpleBindCredentials.dn = ldap_simple_bind_credentials['dn']
        service_account.simpleBindCredentials.password = ldap_simple_bind_credentials['password']
        return service_account
    raise exceptions.Error('Unknown service account type. Supported types are: simpleBindCredentials')