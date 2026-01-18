from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def disable_svc_notifications(self, host, services=None):
    """
        This command is used to prevent notifications from being sent
        out for the specified service.

        Note that this command does not disable notifications from
        being sent out about the host.

        Syntax: DISABLE_SVC_NOTIFICATIONS;<host_name>;<service_description>
        """
    cmd = 'DISABLE_SVC_NOTIFICATIONS'
    if services is None:
        services = []
    for service in services:
        notif_str = self._fmt_notif_str(cmd, host, svc=service)
        self._write_command(notif_str)