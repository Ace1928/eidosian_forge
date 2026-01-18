from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_network_interfaces(eg_launchspec, enis):
    if enis is not None:
        eg_enis = []
        for eni in enis:
            eg_eni = expand_fields(eni_fields, eni, 'NetworkInterface')
            eg_pias = expand_list(eni.get('private_ip_addresses'), private_ip_fields, 'PrivateIpAddress')
            if eg_pias is not None:
                eg_eni.private_ip_addresses = eg_pias
            eg_enis.append(eg_eni)
        if len(eg_enis) > 0:
            eg_launchspec.network_interfaces = eg_enis