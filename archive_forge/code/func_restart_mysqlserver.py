from __future__ import absolute_import, division, print_function
import time
def restart_mysqlserver(self):
    """
        Restart MySQL Server.
        """
    self.log('Restart MySQL Server instance {0}'.format(self.name))
    try:
        response = self.mysql_client.servers.begin_restart(resource_group_name=self.resource_group, server_name=self.name)
    except Exception as exc:
        self.fail('Error restarting mysql server {0} - {1}'.format(self.name, str(exc)))
    return True