from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def get_current_server_registry(self):
    reg = {}
    oem_network_attributes = self.module.params.get('oem_network_attributes')
    network_attributes = self.module.params.get('network_attributes')
    firm_ver = get_idrac_firmware_version(self.idrac)
    if oem_network_attributes:
        if LooseVersion(firm_ver) >= '6.0':
            reg = get_dynamic_uri(self.idrac, self.oem_uri, 'Attributes')
        elif '3.0' < LooseVersion(firm_ver) < '6.0':
            reg = self.__get_registry_fw_less_than_6_more_than_3()
        else:
            reg = self.__get_registry_fw_less_than_3()
    if network_attributes:
        resp = get_dynamic_uri(self.idrac, self.redfish_uri)
        reg.update({'Ethernet': resp.get('Ethernet', {})})
        reg.update({'FibreChannel': resp.get('FibreChannel', {})})
        reg.update({'iSCSIBoot': resp.get('iSCSIBoot', {})})
    return reg