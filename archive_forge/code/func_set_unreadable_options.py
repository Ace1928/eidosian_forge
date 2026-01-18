from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def set_unreadable_options(opts, states):
    states['admin_pass'] = opts.get('admin_pass')
    states['eip_id'] = opts.get('eip_id')
    set_unread_nics(opts.get('nics'), states.get('nics'))
    set_unread_root_volume(opts.get('root_volume'), states.get('root_volume'))
    states['security_groups'] = opts.get('security_groups')
    states['server_metadata'] = opts.get('server_metadata')