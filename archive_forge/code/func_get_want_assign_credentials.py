from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_want_assign_credentials(self, AssignCredentials):
    """
        Get the Credentials to be assigned to a site from the playbook.
        Check this API using the check_return_status.

        Parameters:
            AssignCredentials (dict) - Playbook details containing
            credentials that need to be assigned to a site.

        Returns:
            self - The current object with updated information of credentials
            that need to be assigned to a site from the playbook.
        """
    want = {'assign_credentials': {}}
    site_name = AssignCredentials.get('site_name')
    if not site_name:
        self.msg = 'site_name is required for AssignCredentials'
        self.status = 'failed'
        return self
    site_id = []
    for site_name in site_name:
        siteId = self.get_site_id(site_name)
        if not site_name:
            self.msg = 'site_name is invalid in AssignCredentials'
            self.status = 'failed'
            return self
        site_id.append(siteId)
    want.update({'site_id': site_id})
    global_credentials = self.get_global_credentials_params()
    cli_credential = AssignCredentials.get('cli_credential')
    if cli_credential:
        cliId = cli_credential.get('id')
        cliDescription = cli_credential.get('description')
        cliUsername = cli_credential.get('username')
        if cliId or (cliDescription and cliUsername):
            cli_details = global_credentials.get('cliCredential')
            if not cli_details:
                self.msg = 'Global CLI credential is not available'
                self.status = 'failed'
                return self
            cliDetail = None
            if cliId:
                cliDetail = get_dict_result(cli_details, 'id', cliId)
                if not cliDetail:
                    self.msg = 'The ID for the CLI credential is not valid.'
                    self.status = 'failed'
                    return self
            elif cliDescription and cliUsername:
                for item in cli_details:
                    if item.get('description') == cliDescription and item.get('username') == cliUsername:
                        cliDetail = item
                if not cliDetail:
                    self.msg = 'The username and description of the CLI credential are invalid'
                    self.status = 'failed'
                    return self
            want.get('assign_credentials').update({'cliId': cliDetail.get('id')})
    snmp_v2c_read = AssignCredentials.get('snmp_v2c_read')
    if snmp_v2c_read:
        snmpV2cReadId = snmp_v2c_read.get('id')
        snmpV2cReadDescription = snmp_v2c_read.get('description')
        if snmpV2cReadId or snmpV2cReadDescription:
            snmpV2cRead_details = global_credentials.get('snmpV2cRead')
            if not snmpV2cRead_details:
                self.msg = 'Global snmpV2cRead credential is not available'
                self.status = 'failed'
                return self
            snmpV2cReadDetail = None
            if snmpV2cReadId:
                snmpV2cReadDetail = get_dict_result(snmpV2cRead_details, 'id', snmpV2cReadId)
                if not snmpV2cReadDetail:
                    self.msg = 'The ID of the snmpV2cRead credential is not valid.'
                    self.status = 'failed'
                    return self
            elif snmpV2cReadDescription:
                for item in snmpV2cRead_details:
                    if item.get('description') == snmpV2cReadDescription:
                        snmpV2cReadDetail = item
                if not snmpV2cReadDetail:
                    self.msg = 'The username and description for the snmpV2cRead credential are invalid.'
                    self.status = 'failed'
                    return self
            want.get('assign_credentials').update({'snmpV2ReadId': snmpV2cReadDetail.get('id')})
    snmp_v2c_write = AssignCredentials.get('snmp_v2c_write')
    if snmp_v2c_write:
        snmpV2cWriteId = snmp_v2c_write.get('id')
        snmpV2cWriteDescription = snmp_v2c_write.get('description')
        if snmpV2cWriteId or snmpV2cWriteDescription:
            snmpV2cWrite_details = global_credentials.get('snmpV2cWrite')
            if not snmpV2cWrite_details:
                self.msg = 'Global snmpV2cWrite Credential is not available'
                self.status = 'failed'
                return self
            snmpV2cWriteDetail = None
            if snmpV2cWriteId:
                snmpV2cWriteDetail = get_dict_result(snmpV2cWrite_details, 'id', snmpV2cWriteId)
                if not snmpV2cWriteDetail:
                    self.msg = 'The ID of the snmpV2cWrite credential is invalid.'
                    self.status = 'failed'
                    return self
            elif snmpV2cWriteDescription:
                for item in snmpV2cWrite_details:
                    if item.get('description') == snmpV2cWriteDescription:
                        snmpV2cWriteDetail = item
                if not snmpV2cWriteDetail:
                    self.msg = 'The username and description of the snmpV2cWrite credential are invalid.'
                    self.status = 'failed'
                    return self
            want.get('assign_credentials').update({'snmpV2WriteId': snmpV2cWriteDetail.get('id')})
    https_read = AssignCredentials.get('https_read')
    if https_read:
        httpReadId = https_read.get('id')
        httpReadDescription = https_read.get('description')
        httpReadUsername = https_read.get('username')
        if httpReadId or (httpReadDescription and httpReadUsername):
            httpRead_details = global_credentials.get('httpsRead')
            if not httpRead_details:
                self.msg = 'Global httpRead Credential is not available.'
                self.status = 'failed'
                return self
            httpReadDetail = None
            if httpReadId:
                httpReadDetail = get_dict_result(httpRead_details, 'id', httpReadId)
                if not httpReadDetail:
                    self.msg = 'The ID of the httpRead credential is not valid.'
                    self.status = 'failed'
                    return self
            elif httpReadDescription and httpReadUsername:
                for item in httpRead_details:
                    if item.get('description') == httpReadDescription and item.get('username') == httpReadUsername:
                        httpReadDetail = item
                if not httpReadDetail:
                    self.msg = 'The description and username for the httpRead credential are invalid.'
                    self.status = 'failed'
                    return self
            want.get('assign_credentials').update({'httpRead': httpReadDetail.get('id')})
    https_write = AssignCredentials.get('https_write')
    if https_write:
        httpWriteId = https_write.get('id')
        httpWriteDescription = https_write.get('description')
        httpWriteUsername = https_write.get('username')
        if httpWriteId or (httpWriteDescription and httpWriteUsername):
            httpWrite_details = global_credentials.get('httpsWrite')
            if not httpWrite_details:
                self.msg = 'Global httpWrite credential is not available.'
                self.status = 'failed'
                return self
            httpWriteDetail = None
            if httpWriteId:
                httpWriteDetail = get_dict_result(httpWrite_details, 'id', httpWriteId)
                if not httpWriteDetail:
                    self.msg = 'The ID of the httpWrite credential is not valid.'
                    self.status = 'failed'
                    return self
            elif httpWriteDescription and httpWriteUsername:
                for item in httpWrite_details:
                    if item.get('description') == httpWriteDescription and item.get('username') == httpWriteUsername:
                        httpWriteDetail = item
                if not httpWriteDetail:
                    self.msg = 'The description and username for the httpWrite credential are invalid.'
                    self.status = 'failed'
                    return self
            want.get('assign_credentials').update({'httpWrite': httpWriteDetail.get('id')})
    snmp_v3 = AssignCredentials.get('snmp_v3')
    if snmp_v3:
        snmpV3Id = snmp_v3.get('id')
        snmpV3Description = snmp_v3.get('description')
        if snmpV3Id or snmpV3Description:
            snmpV3_details = global_credentials.get('snmpV3')
            if not snmpV3_details:
                self.msg = 'Global snmpV3 Credential is not available.'
                self.status = 'failed'
                return self
            snmpV3Detail = None
            if snmpV3Id:
                snmpV3Detail = get_dict_result(snmpV3_details, 'id', snmpV3Id)
                if not snmpV3Detail:
                    self.msg = 'The ID of the snmpV3 credential is not valid.'
                    self.status = 'failed'
                    return self
            elif snmpV3Description:
                for item in snmpV3_details:
                    if item.get('description') == snmpV3Description:
                        snmpV3Detail = item
                if not snmpV3Detail:
                    self.msg = 'The username and description for the snmpV2cWrite credential are invalid.'
                    self.status = 'failed'
                    return self
            want.get('assign_credentials').update({'snmpV3Id': snmpV3Detail.get('id')})
    self.log('Desired State (want): {0}'.format(want), 'INFO')
    self.want.update(want)
    self.msg = 'Collected the Credentials needed to be assigned from the Cisco DNA Center'
    self.status = 'success'
    return self