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
def start_challenges(self):
    """
        Create new authorizations for all identifiers of the CSR,
        respectively start a new order for ACME v2.
        """
    self.authorizations = {}
    if self.version == 1:
        for identifier_type, identifier in self.identifiers:
            if identifier_type != 'dns':
                raise ModuleFailException('ACME v1 only supports DNS identifiers!')
        for identifier_type, identifier in self.identifiers:
            authz = Authorization.create(self.client, identifier_type, identifier)
            self.authorizations[authz.combined_identifier] = authz
    else:
        self.order = Order.create(self.client, self.identifiers)
        self.order_uri = self.order.url
        self.order.load_authorizations(self.client)
        self.authorizations.update(self.order.authorizations)
    self.changed = True