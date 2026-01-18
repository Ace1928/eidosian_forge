from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
def get_lun_mapping(self):
    """Find the matching lun mapping reference.

        Returns: tuple(bool, int, int): contains volume match, volume mapping reference and mapping lun
        """
    target_match = False
    reference = None
    lun = None
    self.update_mapping_info()
    if self.lun and any((self.lun == lun_mapping['lun'] and self.target == self.mapping_info['target_by_reference'][lun_mapping['map_reference']] and (self.volume != self.mapping_info['volume_by_reference'][lun_mapping['volume_reference']]) for lun_mapping in self.mapping_info['lun_mapping'])):
        self.module.fail_json(msg='Option lun value is already in use for target! Array Id [%s].' % self.ssid)
    if self.target and self.target_type and (self.target in self.mapping_info['target_type_by_name'].keys()) and (self.mapping_info['target_type_by_name'][self.target] != self.target_type):
        self.module.fail_json(msg='Option target does not match the specified target_type! Id [%s].' % self.ssid)
    if self.state:
        if self.volume not in self.mapping_info['volume_by_name'].keys():
            self.module.fail_json(msg='Volume does not exist. Id [%s].' % self.ssid)
        if self.target and self.target not in self.mapping_info['target_by_name'].keys():
            self.module.fail_json(msg="Target does not exist. Id [%s'." % self.ssid)
    for lun_mapping in self.mapping_info['lun_mapping']:
        if lun_mapping['volume_reference'] == self.mapping_info['volume_by_name'][self.volume]:
            reference = lun_mapping['lun_mapping_reference']
            lun = lun_mapping['lun']
            if lun_mapping['map_reference'] in self.mapping_info['target_by_reference'].keys() and self.mapping_info['target_by_reference'][lun_mapping['map_reference']] == self.target and (self.lun is None or lun == self.lun):
                target_match = True
    return (target_match, reference, lun)