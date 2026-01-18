from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def make_hostgroup(module, array):
    if module.params['rename']:
        module.fail_json(msg='Hostgroup {0} does not exist - rename failed.'.format(module.params['name']))
    changed = True
    if not module.check_mode:
        try:
            array.create_hgroup(module.params['name'])
        except Exception:
            module.fail_json(msg='Failed to create hostgroup {0}'.format(module.params['name']))
        if module.params['host']:
            array.set_hgroup(module.params['name'], hostlist=module.params['host'])
        if module.params['volume']:
            if len(module.params['volume']) == 1 and module.params['lun']:
                try:
                    array.connect_hgroup(module.params['name'], module.params['volume'][0], lun=module.params['lun'])
                except Exception:
                    module.fail_json(msg='Failed to add volume {0} with LUN ID {1}'.format(module.params['volume'][0], module.params['lun']))
            else:
                for vol in module.params['volume']:
                    try:
                        array.connect_hgroup(module.params['name'], vol)
                    except Exception:
                        module.fail_json(msg='Failed to add volume to hostgroup')
    module.exit_json(changed=changed)