from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def enable_host_svc_notifications(self, host):
    """
        Enables notifications for all services on the specified host.

        Note that this does not enable notifications for the host.

        Syntax: ENABLE_HOST_SVC_NOTIFICATIONS;<host_name>
        """
    cmd = 'ENABLE_HOST_SVC_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, host)
    nagios_return = self._write_command(notif_str)
    if nagios_return:
        return notif_str
    else:
        return 'Fail: could not write to the command file'