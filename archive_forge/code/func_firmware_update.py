from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def firmware_update(obj, module):
    """Firmware update using single binary file from Local path or HTTP location."""
    image_path = module.params.get('image_uri')
    trans_proto = module.params['transfer_protocol']
    inventory_uri, push_uri, update_uri = _get_update_service_target(obj, module)
    if image_path.startswith('http'):
        payload = {'ImageURI': image_path, 'TransferProtocol': trans_proto}
        update_status = obj.invoke_request('POST', update_uri, data=payload)
    else:
        resp_inv = obj.invoke_request('GET', inventory_uri)
        with open(os.path.join(image_path), 'rb') as img_file:
            binary_payload = {'file': (image_path.split(os.sep)[-1], img_file, 'multipart/form-data')}
            data, ctype = _encode_form_data(binary_payload)
        headers = {'If-Match': resp_inv.headers.get('etag')}
        headers.update({'Content-Type': ctype})
        upload_status = obj.invoke_request('POST', push_uri, data=data, headers=headers, dump=False, api_timeout=module.params['timeout'])
        if upload_status.status_code == 201:
            payload = {'ImageURI': upload_status.headers.get('location')}
            update_status = obj.invoke_request('POST', update_uri, data=payload)
        else:
            update_status = upload_status
    return update_status