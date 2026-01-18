from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, wait_for_task, get_parent_datacenter
import os
def download_screenshot_file(self, file_url, local_file_path, file_name):
    response = None
    download_size = 0
    if local_file_path.endswith('.png'):
        local_file_name = local_file_path.split('/')[-1]
        local_file_path = local_file_path.rsplit('/', 1)[0]
    else:
        local_file_name = file_name
    if not os.path.exists(local_file_path):
        try:
            os.makedirs(local_file_path)
        except OSError as err:
            self.module.fail_json(msg='Exception caught when create folder %s on local machine, with error %s' % (local_file_path, to_native(err)))
    local_file = os.path.join(local_file_path, local_file_name)
    with open(local_file, 'wb') as handle:
        try:
            response = open_url(file_url, url_username=self.params.get('username'), url_password=self.params.get('password'), validate_certs=False)
        except Exception as err:
            self.module.fail_json(msg='Download screenshot file from URL %s, failed due to %s' % (file_url, to_native(err)))
        if not response or response.getcode() >= 400:
            self.module.fail_json(msg='Download screenshot file from URL %s, failed with response %s, response code %s' % (file_url, response, response.getcode()))
        bytes_read = response.read(2 ** 20)
        while bytes_read:
            handle.write(bytes_read)
            handle.flush()
            os.fsync(handle.fileno())
            download_size += len(bytes_read)
            bytes_read = response.read(2 ** 20)
    return download_size