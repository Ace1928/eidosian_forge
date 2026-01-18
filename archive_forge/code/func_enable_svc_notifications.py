from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def enable_svc_notifications(self, host, services=None):
    """
        Enables notifications for a particular service.

        Note that this does not enable notifications for the host.

        Syntax: ENABLE_SVC_NOTIFICATIONS;<host_name>;<service_description>
        """
    cmd = 'ENABLE_SVC_NOTIFICATIONS'
    if services is None:
        services = []
    nagios_return = True
    return_str_list = []
    for service in services:
        notif_str = self._fmt_notif_str(cmd, host, svc=service)
        nagios_return = self._write_command(notif_str) and nagios_return
        return_str_list.append(notif_str)
    if nagios_return:
        return return_str_list
    else:
        return 'Fail: could not write to the command file'