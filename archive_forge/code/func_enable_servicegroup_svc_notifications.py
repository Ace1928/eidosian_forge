from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def enable_servicegroup_svc_notifications(self, servicegroup):
    """
        Enables notifications for all services that are members of a
        particular servicegroup.

        Note that this does not enable notifications for the hosts in
        this servicegroup.

        Syntax: ENABLE_SERVICEGROUP_SVC_NOTIFICATIONS;<servicegroup_name>
        """
    cmd = 'ENABLE_SERVICEGROUP_SVC_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, servicegroup)
    nagios_return = self._write_command(notif_str)
    if nagios_return:
        return notif_str
    else:
        return 'Fail: could not write to the command file'