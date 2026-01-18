from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url
def send_annotation_event(module, key, msg, annotated_by='Ansible', level=None, instance_id=None, event_epoch=None):
    """Send an annotation event to Stackdriver"""
    annotation_api = 'https://event-gateway.stackdriver.com/v1/annotationevent'
    params = {}
    params['message'] = msg
    if annotated_by:
        params['annotated_by'] = annotated_by
    if level:
        params['level'] = level
    if instance_id:
        params['instance_id'] = instance_id
    if event_epoch:
        params['event_epoch'] = event_epoch
    return do_send_request(module, annotation_api, params, key)