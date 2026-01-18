from __future__ import absolute_import, division, print_function
import time
def create_update_firewallrule(self):
    """
        Creates or updates MariaDB firewall rule with the specified configuration.

        :return: deserialized MariaDB firewall rule instance state dictionary
        """
    self.log('Creating / Updating the MariaDB firewall rule instance {0}'.format(self.name))
    try:
        response = self.mariadb_client.firewall_rules.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, firewall_rule_name=self.name, parameters=self.parameters)
        if isinstance(response, LROPoller):
            response = self.get_poller_result(response)
    except Exception as exc:
        self.log('Error attempting to create the MariaDB firewall rule instance.')
        self.fail('Error creating the MariaDB firewall rule instance: {0}'.format(str(exc)))
    return response.as_dict()