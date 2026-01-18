from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_scp_share_details(module):
    share_name = module.params.get('share_name')
    command = module.params['command']
    scp_file_name_format = get_scp_file_format(module)
    if ':/' in share_name:
        nfs_split = share_name.split(':/', 1)
        share = {'share_ip': nfs_split[0], 'share_name': '/{0}'.format(nfs_split[1]), 'share_type': 'NFS'}
        if command == 'export':
            share['file_name'] = scp_file_name_format
    elif '\\' in share_name:
        cifs_share = share_name.split('\\', 3)
        share_ip = cifs_share[2]
        share_path_name = cifs_share[-1]
        if not any((domain in module.params.get('share_user') for domain in DOMAIN_LIST)):
            module.params['share_user'] = '.\\{0}'.format(module.params.get('share_user'))
        share = {'share_ip': share_ip, 'share_name': share_path_name, 'share_type': 'CIFS', 'username': module.params.get('share_user'), 'password': module.params.get('share_password')}
        if command == 'export':
            share['file_name'] = scp_file_name_format
    else:
        share = {'share_type': 'LOCAL', 'share_name': share_name}
        if command == 'export':
            share['file_name'] = scp_file_name_format
    return (share, scp_file_name_format)