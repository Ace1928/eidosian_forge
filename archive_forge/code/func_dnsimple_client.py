from __future__ import absolute_import, division, print_function
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
def dnsimple_client(self):
    """creates a dnsimple client object"""
    if self.account_email and self.account_api_token:
        client = Client(sandbox=self.sandbox, email=self.account_email, access_token=self.account_api_token, user_agent='ansible/community.general')
    else:
        msg = 'Option account_email or account_api_token not provided. Dnsimple authentication with a .dnsimple config file is not supported with dnsimple-python>=2.0.0'
        raise DNSimpleException(msg)
    client.identity.whoami()
    self.client = client