from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url
def axapi_get_vport_protocol(protocol):
    return AXAPI_VPORT_PROTOCOLS.get(protocol.lower(), None)