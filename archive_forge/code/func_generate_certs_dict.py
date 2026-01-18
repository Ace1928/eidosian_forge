from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_certs_dict(array):
    certs_info = {}
    api_version = array._list_available_rest_versions()
    if P53_API_VERSION in api_version:
        certs = array.list_certificates()
        for cert in range(0, len(certs)):
            certificate = certs[cert]['name']
            valid_from = time.strftime('%a, %d %b %Y %H:%M:%S %Z', time.localtime(certs[cert]['valid_from'] / 1000))
            valid_to = time.strftime('%a, %d %b %Y %H:%M:%S %Z', time.localtime(certs[cert]['valid_to'] / 1000))
            certs_info[certificate] = {'status': certs[cert]['status'], 'issued_to': certs[cert]['issued_to'], 'valid_from': valid_from, 'locality': certs[cert]['locality'], 'country': certs[cert]['country'], 'issued_by': certs[cert]['issued_by'], 'valid_to': valid_to, 'state': certs[cert]['state'], 'key_size': certs[cert]['key_size'], 'org_unit': certs[cert]['organizational_unit'], 'common_name': certs[cert]['common_name'], 'organization': certs[cert]['organization'], 'email': certs[cert]['email']}
    return certs_info