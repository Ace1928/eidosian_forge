from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def find_non_existing_pairs(rcg_pairs, input_pairs):
    for pair in rcg_pairs:
        for input_pair in list(input_pairs):
            if input_pair['source_volume_id'] == pair['localVolumeId'] and input_pair['target_volume_id'] == pair['remoteVolumeId']:
                input_pairs.remove(input_pair)
    return input_pairs