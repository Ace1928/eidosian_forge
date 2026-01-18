from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def attached_volume(self):
    volume = self.present_volume()
    if volume:
        if volume.get('virtualmachineid') != self.get_vm(key='id'):
            self.result['changed'] = True
            if not self.module.check_mode:
                volume = self.detached_volume()
        if 'attached' not in volume:
            self.result['changed'] = True
            args = {'id': volume['id'], 'virtualmachineid': self.get_vm(key='id'), 'deviceid': self.module.params.get('device_id')}
            if not self.module.check_mode:
                res = self.query_api('attachVolume', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    volume = self.poll_job(res, 'volume')
    return volume