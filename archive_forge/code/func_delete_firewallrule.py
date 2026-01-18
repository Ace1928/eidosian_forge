from __future__ import absolute_import, division, print_function
import time
def delete_firewallrule(self):
    """
        Deletes specified MariaDB firewall rule instance in the specified subscription and resource group.

        :return: True
        """
    self.log('Deleting the MariaDB firewall rule instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.firewall_rules.begin_delete(resource_group_name=self.resource_group, server_name=self.server_name, firewall_rule_name=self.name)
    except Exception as e:
        self.log('Error attempting to delete the MariaDB firewall rule instance.')
        self.fail('Error deleting the MariaDB firewall rule instance: {0}'.format(str(e)))
    return True