from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def schedule_servicegroup_host_downtime(self, servicegroup, minutes=30, start=None):
    """
        This command is used to schedule downtime for all hosts in a
        particular servicegroup.

        During the specified downtime, Nagios will not send
        notifications out about the hosts.

        Syntax: SCHEDULE_SERVICEGROUP_HOST_DOWNTIME;<servicegroup_name>;
        <start_time>;<end_time>;<fixed>;<trigger_id>;<duration>;<author>;
        <comment>
        """
    cmd = 'SCHEDULE_SERVICEGROUP_HOST_DOWNTIME'
    dt_cmd_str = self._fmt_dt_str(cmd, servicegroup, minutes, start=start)
    self._write_command(dt_cmd_str)