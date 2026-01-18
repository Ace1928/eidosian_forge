from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def job_config_to_string(xml_str):
    return ET.tostring(ET.fromstring(xml_str)).decode('ascii')