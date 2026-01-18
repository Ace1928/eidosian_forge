from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def create_cvo_azure(self):
    """
        Create AZURE CVO
        """
    if self.parameters.get('workspace_id') is None:
        response, msg = self.na_helper.get_tenant(self.rest_api, self.headers)
        if response is None:
            self.module.fail_json(msg)
        self.parameters['workspace_id'] = response
    if self.parameters.get('nss_account') is None:
        if self.parameters.get('serial_number') is not None:
            if not self.parameters['serial_number'].startswith('Eval-') and self.parameters['license_type'] == 'azure-cot-premium-byol':
                response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                if response is None:
                    self.module.fail_json(msg)
                self.parameters['nss_account'] = response
        elif self.parameters.get('platform_serial_number_node1') is not None and self.parameters.get('platform_serial_number_node2') is not None:
            if not self.parameters['platform_serial_number_node1'].startswith('Eval-') and (not self.parameters['platform_serial_number_node2'].startswith('Eval-')) and (self.parameters['license_type'] == 'azure-ha-cot-premium-byol'):
                response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                if response is None:
                    self.module.fail_json(msg)
                self.parameters['nss_account'] = response
    json = {'name': self.parameters['name'], 'region': self.parameters['location'], 'subscriptionId': self.parameters['subscription_id'], 'tenantId': self.parameters['workspace_id'], 'storageType': self.parameters['storage_type'], 'dataEncryptionType': self.parameters['data_encryption_type'], 'optimizedNetworkUtilization': True, 'diskSize': {'size': self.parameters['disk_size'], 'unit': self.parameters['disk_size_unit']}, 'svmPassword': self.parameters['svm_password'], 'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs'], 'enableCompliance': self.parameters['enable_compliance'], 'enableMonitoring': self.parameters['enable_monitoring'], 'vsaMetadata': {'ontapVersion': self.parameters['ontap_version'], 'licenseType': self.parameters['license_type'], 'useLatestVersion': self.parameters['use_latest_version'], 'instanceType': self.parameters['instance_type']}}
    if self.parameters['capacity_tier'] == 'Blob':
        json.update({'capacityTier': self.parameters['capacity_tier'], 'tierLevel': self.parameters['tier_level']})
    if self.parameters.get('provided_license') is not None:
        json['vsaMetadata'].update({'providedLicense': self.parameters['provided_license']})
    if not self.parameters['license_type'].endswith('capacity-paygo'):
        json['vsaMetadata'].update({'capacityPackageName': ''})
    if self.parameters.get('capacity_package_name') is not None:
        json['vsaMetadata'].update({'capacityPackageName': self.parameters['capacity_package_name']})
    if self.parameters.get('cidr') is not None:
        json.update({'cidr': self.parameters['cidr']})
    if self.parameters.get('writing_speed_state') is not None:
        json.update({'writingSpeedState': self.parameters['writing_speed_state'].upper()})
    if self.parameters.get('resource_group') is not None:
        json.update({'resourceGroup': self.parameters['resource_group'], 'allowDeployInExistingRg': self.parameters['allow_deploy_in_existing_rg']})
    else:
        json.update({'resourceGroup': self.parameters['name'] + '-rg'})
    if self.parameters.get('serial_number') is not None:
        json.update({'serialNumber': self.parameters['serial_number']})
    if self.parameters.get('security_group_id') is not None:
        json.update({'securityGroupId': self.parameters['security_group_id']})
    if self.parameters.get('cloud_provider_account') is not None:
        json.update({'cloudProviderAccount': self.parameters['cloud_provider_account']})
    if self.parameters.get('backup_volumes_to_cbs') is not None:
        json.update({'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs']})
    if self.parameters.get('nss_account') is not None:
        json.update({'nssAccount': self.parameters['nss_account']})
    if self.parameters.get('availability_zone') is not None:
        json.update({'availabilityZone': self.parameters['availability_zone']})
    if self.parameters['data_encryption_type'] == 'AZURE':
        if self.parameters.get('azure_encryption_parameters') is not None:
            json.update({'azureEncryptionParameters': {'key': self.parameters['azure_encryption_parameters']}})
    if self.parameters.get('svm_name') is not None:
        json.update({'svmName': self.parameters['svm_name']})
    if self.parameters.get('azure_tag') is not None:
        tags = []
        for each_tag in self.parameters['azure_tag']:
            tag = {'tagKey': each_tag['tag_key'], 'tagValue': each_tag['tag_value']}
            tags.append(tag)
        json.update({'azureTags': tags})
    if self.parameters['is_ha']:
        ha_params = dict()
        if self.parameters.get('platform_serial_number_node1'):
            ha_params['platformSerialNumberNode1'] = self.parameters['platform_serial_number_node1']
        if self.parameters.get('platform_serial_number_node2'):
            ha_params['platformSerialNumberNode2'] = self.parameters['platform_serial_number_node2']
        if self.parameters.get('availability_zone_node1'):
            ha_params['availabilityZoneNode1'] = self.parameters['availability_zone_node1']
        if self.parameters.get('availability_zone_node2'):
            ha_params['availabilityZoneNode2'] = self.parameters['availability_zone_node2']
        if self.parameters.get('ha_enable_https') is not None:
            ha_params['enableHttps'] = self.parameters['ha_enable_https']
        json['haParams'] = ha_params
    resource_group = self.parameters['vnet_resource_group'] if self.parameters.get('vnet_resource_group') is not None else self.parameters['resource_group']
    resource_group_path = 'subscriptions/%s/resourceGroups/%s' % (self.parameters['subscription_id'], resource_group)
    vnet_format = '%s/%s' if self.rest_api.simulator else '/%s/providers/Microsoft.Network/virtualNetworks/%s'
    vnet = vnet_format % (resource_group_path, self.parameters['vnet_id'])
    json.update({'vnetId': vnet})
    json.update({'subnetId': '%s/subnets/%s' % (vnet, self.parameters['subnet_id'])})
    api_url = '%s/working-environments' % self.rest_api.api_root_path
    response, error, on_cloud_request_id = self.rest_api.post(api_url, json, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on creating cvo azure: %s, %s' % (str(error), str(response)))
    working_environment_id = response['publicId']
    wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
    err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'create', 60, 60)
    if err is not None:
        self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating CVO AZURE: %s' % str(err))
    return working_environment_id