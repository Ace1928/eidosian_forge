from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.network import (
def md5sum_check(self, dst, file_system):
    command = 'show file {0}{1} md5sum'.format(file_system, dst)
    remote_filehash = self._connection.run_commands(command)[0]
    remote_filehash = to_bytes(remote_filehash, errors='surrogate_or_strict')
    local_file = self._module.params['local_file']
    try:
        with open(local_file, 'rb') as f:
            filecontent = f.read()
    except (OSError, IOError) as exc:
        self._module.fail_json('Error reading the file: {0}'.format(to_text(exc)))
    filecontent = to_bytes(filecontent, errors='surrogate_or_strict')
    local_filehash = hashlib.md5(filecontent).hexdigest()
    decoded_rhash = remote_filehash.decode('UTF-8')
    if local_filehash == decoded_rhash:
        return True
    else:
        return False