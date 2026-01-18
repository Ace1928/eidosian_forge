from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def source_vol_relationship(self, volume):
    """
        Check if the source volume is associated to any migration relationship.
        Returns:
            None if no matching instances
        """
    source_vdisk_data, target_vdisk_data = self.get_existing_vdisk()
    if not source_vdisk_data:
        msg = 'Source volume [%s] does not exist' % self.source_volume
        self.module.exit_json(msg=msg)
    self.log('Trying to get the remote copy relationship')
    relationship_name = source_vdisk_data[0]['RC_name']
    if not relationship_name:
        self.module.fail_json(msg='Volume [%s] cannot be deleted. No Migration relationship is configured with the volume.' % self.source_volume)
    existing_rel_data = self.restapi.svc_obj_info(cmd='lsrcrelationship', cmdopts=None, cmdargs=[relationship_name])
    if existing_rel_data['copy_type'] != 'migration':
        self.module.fail_json(msg='Volume [%s] cannot be deleted. No Migration relationship is configured with the volume.' % self.source_volume)