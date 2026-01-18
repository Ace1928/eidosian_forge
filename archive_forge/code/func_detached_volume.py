from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def detached_volume(self):
    volume = self.present_volume()
    if volume:
        if 'attached' not in volume:
            return volume
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('detachVolume', id=volume['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                volume = self.poll_job(res, 'volume')
    return volume