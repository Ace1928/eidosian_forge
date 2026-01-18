from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def probe_volume(self, data):
    props = {}
    if self.iogrp:
        input_iogrp = set(self.iogrp)
        existing_iogrp = set(self.get_existing_iogrp())
        if input_iogrp ^ existing_iogrp:
            iogrp_to_add = input_iogrp - existing_iogrp
            iogrp_to_remove = existing_iogrp - input_iogrp
            if iogrp_to_add:
                props['iogrp'] = {'add': list(iogrp_to_add)}
            if iogrp_to_remove:
                props['iogrp'] = {'remove': list(iogrp_to_remove)}
    if self.size:
        input_size = self.convert_to_bytes()
        existing_size = int(data[0]['capacity'])
        if input_size != existing_size:
            if input_size > existing_size:
                props['size'] = {'expand': input_size - existing_size}
            elif existing_size > input_size:
                props['size'] = {'shrink': existing_size - input_size}
    if self.volumegroup:
        if self.volumegroup != data[0]['volume_group_name']:
            props['volumegroup'] = {'name': self.volumegroup}
    if self.novolumegroup:
        if data[0]['volume_group_name']:
            props['novolumegroup'] = {'status': True}
    if self.thin is True:
        if data[0]['capacity'] == data[1]['real_capacity'] or data[1]['compressed_copy'] == 'yes':
            props['thin'] = {'status': True}
    if self.compressed is True:
        if data[1]['compressed_copy'] == 'no':
            props['compressed'] = {'status': True}
    if self.deduplicated is True:
        if data[1]['deduplicated_copy'] == 'no':
            props['deduplicated'] = {'status': True}
    if self.pool:
        if self.pool != data[0]['mdisk_grp_name']:
            props['pool'] = {'status': True}
    if self.enable_cloud_snapshot is True:
        if not strtobool(data[0].get('cloud_backup_enabled')):
            props['cloud_backup'] = {'status': True}
    elif self.enable_cloud_snapshot is False:
        if strtobool(data[0].get('cloud_backup_enabled')):
            props['cloud_backup'] = {'status': True}
    if self.cloud_account_name:
        if self.cloud_account_name != data[0].get('cloud_account_name'):
            props['cloud_backup'] = {'status': True}
    return props