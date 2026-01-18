from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def disable_host_notifications(self, host):
    """
        This command is used to prevent notifications from being sent
        out for the specified host.

        Note that this command does not disable notifications for
        services associated with this host.

        Syntax: DISABLE_HOST_NOTIFICATIONS;<host_name>
        """
    cmd = 'DISABLE_HOST_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, host)
    self._write_command(notif_str)