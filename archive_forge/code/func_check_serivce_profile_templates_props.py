from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def check_serivce_profile_templates_props(ucs, module, mo, dn):
    props_match = False
    kwargs = dict(bios_profile_name=module.params['bios_policy'])
    kwargs['boot_policy_name'] = module.params['boot_policy']
    kwargs['descr'] = module.params['description']
    kwargs['ext_ip_state'] = module.params['mgmt_ip_state']
    kwargs['ext_ip_pool_name'] = module.params['mgmt_ip_pool']
    kwargs['host_fw_policy_name'] = module.params['host_firmware_package']
    kwargs['ident_pool_name'] = module.params['uuid_pool']
    kwargs['kvm_mgmt_policy_name'] = module.params['kvm_mgmt_policy']
    kwargs['local_disk_policy_name'] = module.params['local_disk_policy']
    kwargs['maint_policy_name'] = module.params['maintenance_policy']
    kwargs['mgmt_access_policy_name'] = module.params['ipmi_access_profile']
    kwargs['power_policy_name'] = module.params['power_control_policy']
    kwargs['power_sync_policy_name'] = module.params['power_sync_policy']
    kwargs['scrub_policy_name'] = module.params['scrub_policy']
    kwargs['sol_policy_name'] = module.params['sol_policy']
    kwargs['stats_policy_name'] = module.params['threshold_policy']
    kwargs['type'] = module.params['template_type']
    kwargs['usr_lbl'] = module.params['user_label']
    kwargs['vmedia_policy_name'] = module.params['vmedia_policy']
    if mo.check_prop_match(**kwargs):
        props_match = check_storage_profile_props(ucs, module, dn)
        if props_match:
            props_match = check_connectivity_policy_props(ucs, module, dn)
        if props_match:
            props_match = check_iqn_pool_props(ucs, module, dn)
        if props_match:
            props_match = check_inband_management_props(ucs, module, dn)
        if props_match:
            props_match = check_power_props(ucs, module, dn)
        if props_match:
            props_match = check_server_pool(ucs, module, dn)
    return props_match