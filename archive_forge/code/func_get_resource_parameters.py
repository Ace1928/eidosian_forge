from __future__ import (absolute_import, division, print_function)
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
def get_resource_parameters(module):
    command = module.params['command']
    csr_uri = 'ApplicationService/Actions/ApplicationService.{0}'
    method = 'POST'
    if command == 'generate_csr':
        uri = csr_uri.format('GenerateCSR')
        payload = {'DistinguishedName': module.params['distinguished_name'], 'DepartmentName': module.params['department_name'], 'BusinessName': module.params['business_name'], 'Locality': module.params['locality'], 'State': module.params['country_state'], 'Country': module.params['country'], 'Email': module.params['email'], 'San': get_san(module.params['subject_alternative_names'])}
    else:
        file_path = module.params['upload_file']
        uri = csr_uri.format('UploadCertificate')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as payload:
                payload = payload.read()
        else:
            module.fail_json(msg='No such file or directory.')
    return (method, uri, payload)