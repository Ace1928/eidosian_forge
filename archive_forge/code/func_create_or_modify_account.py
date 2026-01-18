from __future__ import (absolute_import, division, print_function)
import json
import re
import time
from ssl import SSLError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def create_or_modify_account(module, idrac, slot_uri, slot_id, empty_slot_id, empty_slot_uri, user_attr):
    """
    This function create user account in case not exists else update it.
    :param module: user account module arguments
    :param idrac: idrac object
    :param slot_uri: slot uri for update
    :param slot_id: slot id for update
    :param empty_slot_id: empty slot id for create
    :param empty_slot_uri: empty slot uri for create
    :return: json
    """
    generation, firmware_version = idrac.get_server_generation
    msg, response = ('Unable to retrieve the user details.', {})
    if (slot_id and slot_uri) is None and (empty_slot_id and empty_slot_uri) is not None:
        msg = 'Successfully created user account.'
        payload = get_payload(module, empty_slot_id, action='create')
        if module.check_mode:
            module.exit_json(msg='Changes found to commit!', changed=True)
        if generation >= 14:
            response = idrac.invoke_request(ATTRIBUTE_URI, 'PATCH', data={'Attributes': payload})
        elif generation < 14:
            xml_payload, json_payload = convert_payload_xml(payload)
            time.sleep(10)
            response = idrac.import_scp(import_buffer=xml_payload, target='ALL', job_wait=True)
    elif (slot_id and slot_uri) is not None:
        msg = 'Successfully updated user account.'
        payload = get_payload(module, slot_id, action='update')
        xml_payload, json_payload = convert_payload_xml(payload)
        value = compare_payload(json_payload, user_attr)
        if module.check_mode:
            if value:
                module.exit_json(msg='Changes found to commit!', changed=True)
            module.exit_json(msg='No changes found to commit!')
        if not value:
            module.exit_json(msg='Requested changes are already present in the user slot.')
        if generation >= 14:
            response = idrac.invoke_request(ATTRIBUTE_URI, 'PATCH', data={'Attributes': payload})
        elif generation < 14:
            time.sleep(10)
            response = idrac.import_scp(import_buffer=xml_payload, target='ALL', job_wait=True)
    elif (slot_id and slot_uri and empty_slot_id and empty_slot_uri) is None:
        module.fail_json(msg='Maximum number of users reached. Delete a user account and retry the operation.')
    return (response, msg)