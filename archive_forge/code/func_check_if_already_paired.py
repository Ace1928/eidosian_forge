from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def check_if_already_paired(self, vol_id):
    """
            Check for idempotency
            A volume can have only one pair
            Return paired-volume-id if volume is paired already
            None if volume is not paired
        """
    paired_volumes = self.elem.list_volumes(volume_ids=[vol_id], is_paired=True)
    for vol in paired_volumes.volumes:
        for pair in vol.volume_pairs:
            if pair is not None:
                return pair.remote_volume_id
    return None