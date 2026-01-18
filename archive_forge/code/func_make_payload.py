from __future__ import (absolute_import, division, print_function)
import json
import os
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.common.dict_transformations import recursive_diff
def make_payload(prm):
    dc_type = {'DNS': 'DnsServer', 'MANUAL': 'ServerName'}
    tmplt_ad = {'name': 'Name', 'domain_controller_port': 'ServerPort', 'domain_controller_lookup': 'ServerType', 'domain_server': dc_type[prm.get('domain_controller_lookup')], 'group_domain': 'GroupDomain', 'network_timeout': 'NetworkTimeOut', 'search_timeout': 'SearchTimeOut', 'validate_certificate': 'CertificateValidation'}
    payload = dict([(v, prm.get(k)) for k, v in tmplt_ad.items() if prm.get(k) is not None])
    return payload