from __future__ import absolute_import, division, print_function
import base64
import hashlib
import json
import re
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
def get_validation_data(self, client, identifier_type, identifier):
    token = re.sub('[^A-Za-z0-9_\\-]', '_', self.token)
    key_authorization = create_key_authorization(client, token)
    if self.type == 'http-01':
        return {'resource': '.well-known/acme-challenge/{token}'.format(token=token), 'resource_value': key_authorization}
    if self.type == 'dns-01':
        if identifier_type != 'dns':
            return None
        resource = '_acme-challenge'
        value = nopad_b64(hashlib.sha256(to_bytes(key_authorization)).digest())
        record = resource + identifier[1:] if identifier.startswith('*.') else '{0}.{1}'.format(resource, identifier)
        return {'resource': resource, 'resource_value': value, 'record': record}
    if self.type == 'tls-alpn-01':
        if identifier_type == 'ip':
            resource = ipaddress.ip_address(identifier).reverse_pointer
            if not resource.endswith('.'):
                resource += '.'
        else:
            resource = identifier
        value = base64.b64encode(hashlib.sha256(to_bytes(key_authorization)).digest())
        return {'resource': resource, 'resource_original': combine_identifier(identifier_type, identifier), 'resource_value': value}
    return None