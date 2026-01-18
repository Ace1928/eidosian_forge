from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def schedule_hostgroup_svc_downtime(self, hostgroup, minutes=30, start=None):
    """
        This command is used to schedule downtime for all services in
        a particular hostgroup.

        During the specified downtime, Nagios will not send
        notifications out about the services.

        Note that scheduling downtime for services does not
        automatically schedule downtime for the hosts those services
        are associated with.

        Syntax: SCHEDULE_HOSTGROUP_SVC_DOWNTIME;<hostgroup_name>;<start_time>;
        <end_time>;<fixed>;<trigger_id>;<duration>;<author>;<comment>
        """
    cmd = 'SCHEDULE_HOSTGROUP_SVC_DOWNTIME'
    dt_cmd_str = self._fmt_dt_str(cmd, hostgroup, minutes, start=start)
    self._write_command(dt_cmd_str)