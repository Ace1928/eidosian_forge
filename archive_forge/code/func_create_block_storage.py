from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def create_block_storage(self):
    volume_name = self.get_key_or_fail('volume_name')
    snapshot_id = self.module.params['snapshot_id']
    if snapshot_id:
        self.module.params['block_size'] = None
        self.module.params['region'] = None
        block_size = None
        region = None
    else:
        block_size = self.get_key_or_fail('block_size')
        region = self.get_key_or_fail('region')
    description = self.module.params['description']
    data = {'size_gigabytes': block_size, 'name': volume_name, 'description': description, 'region': region, 'snapshot_id': snapshot_id}
    response = self.rest.post('volumes', data=data)
    status = response.status_code
    json = response.json
    if status == 201:
        project_name = self.module.params.get('project_name')
        if project_name:
            urn = 'do:volume:{0}'.format(json['volume']['id'])
            assign_status, error_message, resources = self.projects.assign_to_project(project_name, urn)
            self.module.exit_json(changed=True, id=json['volume']['id'], msg=error_message, assign_status=assign_status, resources=resources)
        else:
            self.module.exit_json(changed=True, id=json['volume']['id'])
    elif status == 409 and json['id'] == 'conflict':
        resized = self.resize_block_storage(volume_name, region, block_size)
        self.module.exit_json(changed=resized)
    else:
        raise DOBlockStorageException(json['message'])