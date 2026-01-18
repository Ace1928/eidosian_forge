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
def check_template_changed(self, template_ids, template_groups, link_templates, clear_templates, template_macros, template_tags, template_content, template_type):
    """Compares template parameters to already existing values if any are found.

        template_json - JSON structures are compared as deep sorted dictionaries,
        template_xml - XML structures are compared as strings, but filtered and formatted first,
        If none above is used, all the other arguments are compared to their existing counterparts
        retrieved from Zabbix API."""
    changed = False
    if template_content is not None and template_type == 'xml':
        existing_template = self.dump_template(template_ids, template_type='xml')
        if self.filter_xml_template(template_content) != self.filter_xml_template(existing_template):
            changed = True
        return changed
    existing_template = self.dump_template(template_ids, template_type='json')
    if template_content is not None and template_type == 'json':
        parsed_template_json = self.load_json_template(template_content)
        if self.diff_template(parsed_template_json, existing_template):
            changed = True
        return changed
    if template_groups is not None:
        if LooseVersion(self._zbx_api_version) >= LooseVersion('6.2'):
            existing_groups = [g['name'] for g in existing_template['zabbix_export']['template_groups']]
        else:
            existing_groups = [g['name'] for g in existing_template['zabbix_export']['groups']]
        if set(template_groups) != set(existing_groups):
            changed = True
    if 'templates' not in existing_template['zabbix_export']['templates'][0]:
        existing_template['zabbix_export']['templates'][0]['templates'] = []
    exist_child_templates = [t['name'] for t in existing_template['zabbix_export']['templates'][0]['templates']]
    if link_templates is not None:
        if set(link_templates) != set(exist_child_templates):
            changed = True
    elif set([]) != set(exist_child_templates):
        changed = True
    if clear_templates is not None:
        for t in clear_templates:
            if t in exist_child_templates:
                changed = True
                break
    if 'macros' not in existing_template['zabbix_export']['templates'][0]:
        existing_template['zabbix_export']['templates'][0]['macros'] = []
    if template_macros is not None:
        existing_macros = existing_template['zabbix_export']['templates'][0]['macros']
        if template_macros != existing_macros:
            changed = True
    if 'tags' not in existing_template['zabbix_export']['templates'][0]:
        existing_template['zabbix_export']['templates'][0]['tags'] = []
    if template_tags is not None:
        existing_tags = existing_template['zabbix_export']['templates'][0]['tags']
        if template_tags != existing_tags:
            changed = True
    return changed