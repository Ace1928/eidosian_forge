from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def disable_hostgroup_svc_notifications(self, hostgroup):
    """
        Disables notifications for all services associated with hosts
        in a particular hostgroup.

        Note that this does not disable notifications for the hosts in
        the hostgroup - see the DISABLE_HOSTGROUP_HOST_NOTIFICATIONS
        command for that.

        Syntax: DISABLE_HOSTGROUP_SVC_NOTIFICATIONS;<hostgroup_name>
        """
    cmd = 'DISABLE_HOSTGROUP_SVC_NOTIFICATIONS'
    notif_str = self._fmt_notif_str(cmd, hostgroup)
    self._write_command(notif_str)