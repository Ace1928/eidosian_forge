from __future__ import annotations
from datetime import datetime, timedelta, timezone
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import FirewallResource
from ..module_utils.vendor.hcloud.servers import (
from ..module_utils.vendor.hcloud.ssh_keys import SSHKey
from ..module_utils.vendor.hcloud.volumes import Volume
def stop_server_if_forced(self):
    previous_server_status = self.hcloud_server.status
    if previous_server_status == Server.STATUS_RUNNING and (not self.module.check_mode):
        if self.module.params.get('force_upgrade') or self.module.params.get('force') or self.module.params.get('state') == 'stopped':
            self.stop_server()
            return previous_server_status
        else:
            self.module.warn(f'You can not upgrade a running instance {self.hcloud_server.name}. You need to stop the instance or use force=true.')
    return None