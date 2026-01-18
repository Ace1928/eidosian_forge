from __future__ import absolute_import, division, print_function
import json
import traceback
import re
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import PY2
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def filter_template(self, template_json):
    keep_keys = set(['graphs', 'templates', 'triggers', 'value_maps'])
    unwanted_keys = set(template_json['zabbix_export']) - keep_keys
    for unwanted_key in unwanted_keys:
        del template_json['zabbix_export'][unwanted_key]
    desc_not_supported = False
    for template in template_json['zabbix_export']['templates']:
        for key in list(template.keys()):
            if not template[key] or (key == 'description' and desc_not_supported):
                template.pop(key)
    return template_json