import base64
import logging
import os
import textwrap
import uuid
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import prettytable
from urllib import error
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient import exc
def build_resource_name(self, event):
    res_name = getattr(event, 'resource_name')

    def get_stack_id():
        if getattr(event, 'stack_id', None) is not None:
            return event.stack_id
        for link in getattr(event, 'links', []):
            if link.get('rel') == 'stack':
                if 'href' not in link:
                    return None
                stack_link = link['href']
                return stack_link.split('/')[-1]
    stack_id = get_stack_id()
    if not stack_id:
        return res_name
    phys_id = getattr(event, 'physical_resource_id', None)
    status = getattr(event, 'resource_status', None)
    is_stack_event = stack_id == phys_id
    if is_stack_event:
        self.id_to_res_info[stack_id] = (stack_id, res_name)
    elif phys_id and status == 'CREATE_IN_PROGRESS':
        self.id_to_res_info[phys_id] = (stack_id, res_name)
    resource_path = []
    if res_name and (not is_stack_event):
        resource_path.append(res_name)
    self.prepend_paths(resource_path, stack_id)
    return '.'.join(resource_path)