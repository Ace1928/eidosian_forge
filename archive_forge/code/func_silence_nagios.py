from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def silence_nagios(self):
    """
        This command is used to disable notifications for all hosts and services
        in nagios.

        This is a 'SHUT UP, NAGIOS' command
        """
    cmd = 'DISABLE_NOTIFICATIONS'
    self._write_command(self._fmt_notif_str(cmd))