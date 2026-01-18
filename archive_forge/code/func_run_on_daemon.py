import json
import os
import os.path
import sys
import tempfile
import ansible.module_utils.basic
from .exceptions import (
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
def run_on_daemon(self):
    result = dict(changed=False, original_message='', message='')
    ttl = os.environ.get('ANSIBLE_TURBO_LOOKUP_TTL', None)
    with ansible_collections.cloud.common.plugins.module_utils.turbo.common.connect(socket_path=self.socket_path(), ttl=ttl) as turbo_socket:
        ansiblez_path = sys.path[0]
        args = self.init_args()
        data = [ansiblez_path, json.dumps(args), dict(os.environ)]
        content = json.dumps(data).encode()
        result = turbo_socket.communicate(content)
    self.exit_json(**result)