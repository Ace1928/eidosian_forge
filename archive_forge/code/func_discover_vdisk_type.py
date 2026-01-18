from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def discover_vdisk_type(self, data):
    self.log('Entering function discover_vdisk_type')
    is_std_mirrored_vol = False
    is_hs_vol = False
    if data[0]['type'] == 'many':
        is_std_mirrored_vol = True
        self.discovered_poolA = data[1]['mdisk_grp_name']
        self.discovered_poolB = data[2]['mdisk_grp_name']
        self.log('The discovered standard mirrored volume "%s" belongs to pools "%s" and "%s"', self.name, self.discovered_poolA, self.discovered_poolB)
    relationship_name = data[0]['RC_name']
    if relationship_name:
        rel_data = self.restapi.svc_obj_info(cmd='lsrcrelationship', cmdopts=None, cmdargs=[relationship_name])
        if rel_data['copy_type'] == 'activeactive':
            is_hs_vol = True
        if is_hs_vol:
            master_vdisk = rel_data['master_vdisk_name']
            aux_vdisk = rel_data['aux_vdisk_name']
            master_vdisk_data = self.restapi.svc_obj_info(cmd='lsvdisk', cmdopts=None, cmdargs=[master_vdisk])
            aux_vdisk_data = self.restapi.svc_obj_info(cmd='lsvdisk', cmdopts=None, cmdargs=[aux_vdisk])
            if is_std_mirrored_vol:
                self.discovered_poolA = master_vdisk_data[1]['mdisk_grp_name']
                self.discovered_poolB = aux_vdisk_data[1]['mdisk_grp_name']
                self.log('The discovered mixed volume "%s" belongs to pools "%s" and "%s"', self.name, self.discovered_poolA, self.discovered_poolB)
            else:
                self.discovered_poolA = master_vdisk_data[0]['mdisk_grp_name']
                self.discovered_poolB = aux_vdisk_data[0]['mdisk_grp_name']
                self.log('The discovered HyperSwap volume "%s" belongs to pools                     "%s" and "%s"', self.name, self.discovered_poolA, self.discovered_poolB)
    if is_std_mirrored_vol and is_hs_vol:
        self.module.fail_json(msg='Unsupported Configuration: Both HyperSwap and Standard Mirror are configured on this volume')
    elif is_hs_vol:
        vdisk_type = 'local hyperswap'
    elif is_std_mirrored_vol:
        vdisk_type = 'standard mirror'
    if not is_std_mirrored_vol and (not is_hs_vol):
        mdisk_grp_name = data[0]['mdisk_grp_name']
        self.discovered_standard_vol_pool = mdisk_grp_name
        vdisk_type = 'standard'
        self.log('The standard volume %s belongs to pool "%s"', self.name, self.discovered_standard_vol_pool)
    return vdisk_type