from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_ldap_group_config(ldap_group_config, msg):
    """Provision FeatureSpec LdapConfig Group from the parsed yaml file.

  Args:
    ldap_group_config: YamlConfigFile, The ldap group data loaded from the yaml
      file given by the user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing the group details of
    a single LDAP auth method for the IdentityServiceFeatureSpec.
  """
    group = msg.IdentityServiceGroupConfig()
    if 'baseDn' not in ldap_group_config:
        raise exceptions.Error('LDAP Authentication method must contain group baseDn.')
    group.baseDn = ldap_group_config['baseDn']
    if 'idAttribute' in ldap_group_config:
        group.idAttribute = ldap_group_config['idAttribute']
    if 'filter' in ldap_group_config:
        group.filter = ldap_group_config['filter']
    return group