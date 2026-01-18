from __future__ import absolute_import, division, print_function
import itertools
import os
from ansible.module_utils.basic import AnsibleModule
def find_mapper_device_name(module, dm_device):
    dmsetup_cmd = module.get_bin_path('dmsetup', True)
    mapper_prefix = '/dev/mapper/'
    rc, dm_name, err = module.run_command('%s info -C --noheadings -o name %s' % (dmsetup_cmd, dm_device))
    if rc != 0:
        module.fail_json(msg='Failed executing dmsetup command.', rc=rc, err=err)
    mapper_device = mapper_prefix + dm_name.rstrip()
    return mapper_device