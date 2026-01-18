from __future__ import absolute_import, division, print_function
import time
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
def load_authorizations(self, client):
    for auth_uri in self.authorization_uris:
        authz = Authorization.from_url(client, auth_uri)
        self.authorizations[authz.combined_identifier] = authz