from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_cli_credentials(self, CredentialDetails, global_credentials):
    """
        Get the current CLI Credential from
        Cisco DNA Center based on the provided playbook details.
        Check this API using the check_return_status.

        Parameters:
            CredentialDetails (dict) - Playbook details containing Global Device Credentials.
            global_credentials (dict) - All global device credentials details.

        Returns:
            cliDetails (List) - The current CLI credentials.
        """
    all_CLI = CredentialDetails.get('cli_credential')
    cli_details = global_credentials.get('cliCredential')
    cliDetails = []
    if all_CLI and cli_details:
        for cliCredential in all_CLI:
            cliDetail = None
            cliId = cliCredential.get('id')
            if cliId:
                cliDetail = get_dict_result(cli_details, 'id', cliId)
                if not cliDetail:
                    self.msg = 'CLI credential ID is invalid'
                    self.status = 'failed'
                    return self
            cliOldDescription = cliCredential.get('old_description')
            cliOldUsername = cliCredential.get('old_username')
            if cliOldDescription and cliOldUsername and (not cliDetail):
                for item in cli_details:
                    if item.get('description') == cliOldDescription and item.get('username') == cliOldUsername:
                        if cliDetail:
                            self.msg = 'More than one CLI credential with same                                             old_description and old_username. Pass ID.'
                            self.status = 'failed'
                            return self
                        cliDetail = item
                if not cliDetail:
                    self.msg = 'CLI credential old_description or old_username is invalid'
                    self.status = 'failed'
                    return self
            cliDescription = cliCredential.get('description')
            cliUsername = cliCredential.get('username')
            if cliDescription and cliUsername and (not cliDetail):
                for item in cli_details:
                    if item.get('description') == cliDescription and item.get('username') == cliUsername:
                        if cliDetail:
                            self.msg = 'More than one CLI Credential with same                                             description and username. Pass ID.'
                            self.status = 'failed'
                            return self
                        cliDetail = item
            cliDetails.append(cliDetail)
    return cliDetails