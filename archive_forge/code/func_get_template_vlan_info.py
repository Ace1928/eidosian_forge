from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ssl import SSLError
def get_template_vlan_info(rest_obj, template_id):
    nic_bonding_tech = ''
    try:
        resp = rest_obj.invoke_request('GET', TEMPLATE_ATTRIBUTE_VIEW.format(template_id=template_id))
        if resp.success:
            nic_model = resp.json_data.get('AttributeGroups', [])
            for xnic in nic_model:
                if xnic.get(KEY_ATTR_NAME) == 'NicBondingTechnology':
                    nic_bonding_list = xnic.get('Attributes', [])
                    for xbnd in nic_bonding_list:
                        if xbnd.get(KEY_ATTR_NAME).lower() == 'nic bonding technology':
                            nic_bonding_tech = xbnd.get('Value')
    except Exception:
        nic_bonding_tech = ''
    return nic_bonding_tech