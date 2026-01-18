from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.acme.acme import (
from ansible_collections.community.crypto.plugins.module_utils.acme.account import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import (
from ansible_collections.community.crypto.plugins.module_utils.acme.orders import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def get_challenges_data(self, first_step):
    """
        Get challenge details for the chosen challenge type.
        Return a tuple of generic challenge details, and specialized DNS challenge details.
        """
    data = {}
    for type_identifier, authz in self.authorizations.items():
        identifier_type, identifier = split_identifier(type_identifier)
        if authz.status == 'valid':
            continue
        data[identifier] = authz.get_challenge_data(self.client)
        if first_step and self.challenge is not None and (self.challenge not in data[identifier]):
            raise ModuleFailException("Found no challenge of type '{0}' for identifier {1}!".format(self.challenge, type_identifier))
    data_dns = {}
    if self.challenge == 'dns-01':
        for identifier, challenges in data.items():
            if self.challenge in challenges:
                values = data_dns.get(challenges[self.challenge]['record'])
                if values is None:
                    values = []
                    data_dns[challenges[self.challenge]['record']] = values
                values.append(challenges[self.challenge]['resource_value'])
    return (data, data_dns)