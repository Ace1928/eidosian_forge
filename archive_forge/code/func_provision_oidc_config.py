from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_oidc_config(auth_method, msg):
    """Provision FeatureSpec OIDCConfig from the parsed yaml file.

  Args:
    auth_method: YamlConfigFile, The data loaded from the yaml file given by the
      user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing a single
      OIDC auth method for the IdentityServiceFeatureSpec.
  """
    if 'name' not in auth_method:
        raise exceptions.Error('OIDC Authentication method must contain name.')
    auth_method_proto = msg.IdentityServiceAuthMethod()
    auth_method_proto.name = auth_method['name']
    oidc_config = auth_method['oidc']
    if 'issuerURI' not in oidc_config or 'clientID' not in oidc_config:
        raise exceptions.Error('input config file OIDC Config must contain issuerURI and clientID.')
    auth_method_proto.oidcConfig = msg.IdentityServiceOidcConfig()
    auth_method_proto.oidcConfig.issuerUri = oidc_config['issuerURI']
    auth_method_proto.oidcConfig.clientId = oidc_config['clientID']
    validate_issuer_uri(auth_method_proto.oidcConfig.issuerUri, auth_method['name'])
    if 'proxy' in auth_method:
        auth_method_proto.proxy = auth_method['proxy']
    if 'certificateAuthorityData' in oidc_config:
        auth_method_proto.oidcConfig.certificateAuthorityData = oidc_config['certificateAuthorityData']
    if 'deployCloudConsoleProxy' in oidc_config:
        auth_method_proto.oidcConfig.deployCloudConsoleProxy = oidc_config['deployCloudConsoleProxy']
    if 'extraParams' in oidc_config:
        auth_method_proto.oidcConfig.extraParams = oidc_config['extraParams']
    if 'groupPrefix' in oidc_config:
        auth_method_proto.oidcConfig.groupPrefix = oidc_config['groupPrefix']
    if 'groupsClaim' in oidc_config:
        auth_method_proto.oidcConfig.groupsClaim = oidc_config['groupsClaim']
    if not auth_method_proto.oidcConfig.groupsClaim and auth_method_proto.oidcConfig.groupPrefix:
        raise exceptions.Error('groupPrefix should be empty for method [{}] because groupsClaim is empty.'.format(auth_method['name']))
    if 'kubectlRedirectURI' in oidc_config:
        auth_method_proto.oidcConfig.kubectlRedirectUri = oidc_config['kubectlRedirectURI']
    if 'scopes' in oidc_config:
        auth_method_proto.oidcConfig.scopes = oidc_config['scopes']
    if 'userClaim' in oidc_config:
        auth_method_proto.oidcConfig.userClaim = oidc_config['userClaim']
    if 'userPrefix' in oidc_config:
        auth_method_proto.oidcConfig.userPrefix = oidc_config['userPrefix']
    if 'clientSecret' in oidc_config:
        auth_method_proto.oidcConfig.clientSecret = oidc_config['clientSecret']
    if 'enableAccessToken' in oidc_config:
        auth_method_proto.oidcConfig.enableAccessToken = oidc_config['enableAccessToken']
    return auth_method_proto