from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def present_volume(self):
    volume = self.get_volume()
    if volume:
        volume = self.update_volume(volume)
    else:
        disk_offering_id = self.get_disk_offering(key='id')
        snapshot_id = self.get_snapshot(key='id')
        if not disk_offering_id and (not snapshot_id):
            self.module.fail_json(msg='Required one of: disk_offering,snapshot')
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'diskofferingid': disk_offering_id, 'displayvolume': self.module.params.get('display_volume'), 'maxiops': self.module.params.get('max_iops'), 'miniops': self.module.params.get('min_iops'), 'projectid': self.get_project(key='id'), 'size': self.module.params.get('size'), 'snapshotid': snapshot_id, 'zoneid': self.get_zone(key='id')}
        if not self.module.check_mode:
            res = self.query_api('createVolume', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                volume = self.poll_job(res, 'volume')
    if volume:
        volume = self.ensure_tags(resource=volume, resource_type='Volume')
        self.volume = volume
    return volume