from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def get_mapping_fields(volume, host_or_cluster):
    """ Get mapping fields """
    luns = host_or_cluster.get_luns()
    for lun in luns:
        if volume.get_name() == lun.volume.get_name():
            field_dict = dict(id=lun.id)
            return field_dict
    return dict()