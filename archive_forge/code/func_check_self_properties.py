from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_self_properties(self, session):
    if self.description is not None:
        session.description = self.description
    if self.encapsulation_vlan_id is not None:
        session.encapsulationVlanId = self.encapsulation_vlan_id
    if self.strip_original_vlan is not None:
        session.stripOriginalVlan = self.strip_original_vlan
    if self.mirrored_packet_length is not None:
        session.mirroredPacketLength = self.mirrored_packet_length
    if self.normal_traffic_allowed is not None:
        session.normalTrafficAllowed = self.normal_traffic_allowed
    if self.sampling_rate is not None:
        session.samplingRate = self.sampling_rate