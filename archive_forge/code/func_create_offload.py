from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def create_offload(module, array):
    """Create offload target"""
    changed = True
    api_version = array.get_rest_version()
    res = array.get_network_interfaces(names=['@offload.data0'])
    if res.status != 200:
        module.fail_json(msg="Offload Network interface doesn't exist. Please resolve.")
    if not list(res.items)[0].enabled:
        module.fail_json(msg='Offload Network interface not correctly configured. Please resolve.')
    if not module.check_mode:
        if module.params['protocol'] == 'gcp':
            if PROFILE_API_VERSION in api_version and module.params['profile']:
                bucket = OffloadGoogleCloud(access_key_id=module.params['access_key'], bucket=module.params['bucket'], secret_access_key=module.params['secret'], profile=module.params['profile'])
            else:
                bucket = OffloadGoogleCloud(access_key_id=module.params['access_key'], bucket=module.params['bucket'], secret_access_key=module.params['secret'])
            offload = OffloadPost(google_cloud=bucket)
        if module.params['protocol'] == 'azure' and module.params['profile']:
            if PROFILE_API_VERSION in api_version:
                bucket = OffloadAzure(container_name=module.params['container'], secret_access_key=module.params['secret'], account_name=module.params['.bucket'], profile=module.params['profile'])
            else:
                bucket = OffloadAzure(container_name=module.params['container'], secret_access_key=module.params['secret'], account_name=module.params['.bucket'])
            offload = OffloadPost(azure=bucket)
        if module.params['protocol'] == 's3' and module.params['profile']:
            if PROFILE_API_VERSION in api_version:
                bucket = OffloadS3(access_key_id=module.params['access_key'], bucket=module.params['bucket'], secret_access_key=module.params['secret'], profile=module.params['profile'])
            else:
                bucket = OffloadS3(access_key_id=module.params['access_key'], bucket=module.params['bucket'], secret_access_key=module.params['secret'])
            offload = OffloadPost(s3=bucket)
        if module.params['protocol'] == 'nfs' and module.params['profile']:
            if PROFILE_API_VERSION in api_version:
                bucket = OffloadNfs(mount_point=module.params['share'], address=module.params['address'], mount_options=module.params['options'], profile=module.params['profile'])
            else:
                bucket = OffloadNfs(mount_point=module.params['share'], address=module.params['address'], mount_options=module.params['options'])
            offload = OffloadPost(nfs=bucket)
        res = array.post_offloads(offload=offload, initialize=module.params['initialize'], names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to create {0} offload {1}. Error: {2}Please perform diagnostic checks.'.format(module.params['protocol'].upper(), module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)