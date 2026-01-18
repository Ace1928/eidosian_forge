from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def copy_file_from_remote(self, local, local_file_directory, file_system):
    cmdroot = 'copy ' + self._module.params['file_pull_protocol'] + '://'
    ruser = self._module.params['remote_scp_server_user'] + '@'
    rserver = self._module.params['remote_scp_server']
    rserverpassword = self._module.params['remote_scp_server_password']
    rfile = self._module.params['remote_file'] + ' '
    if not rfile.startswith('/'):
        rfile = '/' + rfile
    if not self._platform.startswith('DS-') and 'MDS' not in self._model:
        vrf = ' vrf ' + self._module.params['vrf']
    else:
        vrf = ''
    if self._module.params['file_pull_compact']:
        compact = ' compact '
    else:
        compact = ''
    if self._module.params['file_pull_kstack']:
        kstack = ' use-kstack '
    else:
        kstack = ''
    local_dir_root = '/'
    if local_file_directory:
        local_dir_root = self.mkdir(local_file_directory)
    copy_cmd = cmdroot + ruser + rserver + rfile + file_system + local_dir_root + local + compact + vrf + kstack
    self.result['copy_cmd'] = copy_cmd
    pulled = self._connection.pull_file(command=copy_cmd, remotepassword=rserverpassword)
    if pulled:
        self.result['transfer_status'] = 'Received: File copied/pulled to nxos device from remote scp server.'
    else:
        self.result['failed'] = True