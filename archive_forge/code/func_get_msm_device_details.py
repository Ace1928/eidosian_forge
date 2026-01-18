from __future__ import (absolute_import, division, print_function)
import json
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ssl import SSLError
def get_msm_device_details(rest_obj, module):
    """
    Get msm details
    :param rest_obj: session object
    :param module: Ansible module object
    :return: tuple
    1st item: service tag of the domain
    2nd item: msm version of ome-M device
    """
    hostname = get_ip_from_host(module.params['hostname'])
    fabric_design = module.params.get('fabric_design')
    msm_version = ''
    service_tag = get_service_tag_with_fqdn(rest_obj, module)
    domain_details = rest_obj.get_all_items_with_pagination(DOMAIN_URI)
    for each_domain in domain_details['value']:
        if service_tag and service_tag == each_domain['Identifier']:
            msm_version = validate_lead_msm_version(each_domain, module, fabric_design)
            break
        if hostname in each_domain['PublicAddress']:
            msm_version = validate_lead_msm_version(each_domain, module, fabric_design)
            service_tag = each_domain['Identifier']
            break
    else:
        module.fail_json(msg=SYSTEM_NOT_SUPPORTED_ERROR_MSG)
    return (service_tag, msm_version)