from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import urllib3
def provision_google_config(auth_method, msg):
    """Provision FeatureSpec GoogleConfig from the parsed configuration file.

  Args:
    auth_method: YamlConfigFile, The data loaded from the yaml file given by the
      user. YamlConfigFile is from
      googlecloudsdk.command_lib.anthos.common.file_parsers.
    msg: The gkehub messages package.

  Returns:
    member_config: A MemberConfig configuration containing a single Google
    auth method for the IdentityServiceFeatureSpec.
  """
    if 'name' not in auth_method:
        raise exceptions.Error('Google Authentication method must contain name.')
    auth_method_proto = msg.IdentityServiceAuthMethod()
    auth_method_proto.name = auth_method['name']
    google_config = auth_method['google']
    auth_method_proto.googleConfig = msg.IdentityServiceGoogleConfig()
    if 'proxy' in auth_method:
        auth_method_proto.proxy = auth_method['proxy']
    if 'disable' not in google_config:
        raise exceptions.Error('The "disable" field is not set for the authentication method "{}"'.format(auth_method['name']))
    auth_method_proto.googleConfig.disable = google_config['disable']
    return auth_method_proto