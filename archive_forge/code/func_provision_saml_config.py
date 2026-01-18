from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_saml_config(auth_method, msg):
    """Provision FeatureSpec SamlConfig from the parsed configuration file.

  Args:
    auth_method: YamlConfigFile, The data loaded from the yaml file given by the
      user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing a single SAML
    auth method for the IdentityServiceFeatureSpec.
  """
    if 'name' not in auth_method:
        raise exceptions.Error('SAML Authentication method must contain name.')
    auth_method_proto = msg.IdentityServiceAuthMethod()
    auth_method_proto.name = auth_method['name']
    saml_config = auth_method['saml']
    auth_method_proto.samlConfig = msg.IdentityServiceSamlConfig()
    required_fields = ['idpEntityID', 'idpSingleSignOnURI', 'idpCertificateDataList']
    unset_required_fields = [field_name for field_name in required_fields if field_name not in saml_config]
    if unset_required_fields:
        raise exceptions.Error('The following fields are not set for the authentication method {} : {}'.format(auth_method['name'], ', '.join(unset_required_fields)))
    auth_method_proto.samlConfig.identityProviderId = saml_config['idpEntityID']
    auth_method_proto.samlConfig.identityProviderSsoUri = saml_config['idpSingleSignOnURI']
    auth_method_proto.samlConfig.identityProviderCertificates = saml_config['idpCertificateDataList']
    if 'userAttribute' in saml_config:
        auth_method_proto.samlConfig.userAttribute = saml_config['userAttribute']
    if 'groupsAttribute' in saml_config:
        auth_method_proto.samlConfig.groupsAttribute = saml_config['groupsAttribute']
    if 'userPrefix' in saml_config:
        auth_method_proto.samlConfig.userPrefix = saml_config['userPrefix']
    if 'groupPrefix' in saml_config:
        auth_method_proto.samlConfig.groupPrefix = saml_config['groupPrefix']
    if 'attributeMapping' in saml_config:
        auth_method_proto.samlConfig.attributeMapping = msg.IdentityServiceSamlConfig.AttributeMappingValue()
        for attribute_key, attribute_value in saml_config['attributeMapping'].items():
            attribute_map_item = msg.IdentityServiceSamlConfig.AttributeMappingValue.AdditionalProperty()
            attribute_map_item.key = attribute_key
            attribute_map_item.value = attribute_value
            auth_method_proto.samlConfig.attributeMapping.additionalProperties.append(attribute_map_item)
    return auth_method_proto