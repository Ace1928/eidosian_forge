from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def normalize_vswitch_info(vswitch_obj, policy_info):
    """Create vSwitch information"""
    vswitch_info_dict = dict()
    spec = vswitch_obj.spec
    vswitch_info_dict['pnics'] = VswitchInfoManager.serialize_pnics(vswitch_obj)
    vswitch_info_dict['mtu'] = vswitch_obj.mtu
    vswitch_info_dict['num_ports'] = spec.numPorts
    if policy_info:
        if spec.policy.security:
            vswitch_info_dict['security'] = [spec.policy.security.allowPromiscuous, spec.policy.security.macChanges, spec.policy.security.forgedTransmits]
        if spec.policy.shapingPolicy:
            vswitch_info_dict['ts'] = spec.policy.shapingPolicy.enabled
        if spec.policy.nicTeaming:
            vswitch_info_dict['lb'] = spec.policy.nicTeaming.policy
            vswitch_info_dict['notify'] = spec.policy.nicTeaming.notifySwitches
            vswitch_info_dict['failback'] = not spec.policy.nicTeaming.rollingOrder
            vswitch_info_dict['failover_active'] = spec.policy.nicTeaming.nicOrder.activeNic
            vswitch_info_dict['failover_standby'] = spec.policy.nicTeaming.nicOrder.standbyNic
            if spec.policy.nicTeaming.failureCriteria.checkBeacon:
                vswitch_info_dict['failure_detection'] = 'beacon_probing'
            else:
                vswitch_info_dict['failure_detection'] = 'link_status_only'
    return vswitch_info_dict