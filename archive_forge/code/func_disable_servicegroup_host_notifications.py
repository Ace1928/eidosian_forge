from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def disable_servicegroup_host_notifications(self, servicegroup):
    """
        This command is used to prevent notifications from being sent
        out for all hosts in the specified servicegroup.

        Note that this command does not disable notifications for
        services associated with hosts in this service group.

        Syntax: DISABLE_SERVICEGROUP_HOST_NOTIFICATIONS;<servicegroup_name>
        """
    cmd = 'DISABLE_SERVICEGROUP_HOST_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, servicegroup)
    self._write_command(notif_str)