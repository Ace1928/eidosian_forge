from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def create_cvo_aws(self):
    """ Create AWS CVO """
    if self.parameters.get('workspace_id') is None:
        response, msg = self.na_helper.get_tenant(self.rest_api, self.headers)
        if response is None:
            self.module.fail_json(msg)
        self.parameters['workspace_id'] = response
    if self.parameters.get('vpc_id') is None and self.parameters['is_ha'] is False:
        self.parameters['vpc_id'] = self.get_vpc()
    if self.parameters.get('nss_account') is None:
        if self.parameters.get('platform_serial_number') is not None:
            if not self.parameters['platform_serial_number'].startswith('Eval-') and self.parameters['license_type'] == 'cot-premium-byol':
                response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                if response is None:
                    self.module.fail_json(msg)
                self.parameters['nss_account'] = response
        elif self.parameters.get('platform_serial_number_node1') is not None and self.parameters.get('platform_serial_number_node2') is not None:
            if not self.parameters['platform_serial_number_node1'].startswith('Eval-') and (not self.parameters['platform_serial_number_node2'].startswith('Eval-')) and (self.parameters['license_type'] == 'ha-cot-premium-byol'):
                response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                if response is None:
                    self.module.fail_json(msg)
                self.parameters['nss_account'] = response
    json = {'name': self.parameters['name'], 'region': self.parameters['region'], 'tenantId': self.parameters['workspace_id'], 'vpcId': self.parameters['vpc_id'], 'dataEncryptionType': self.parameters['data_encryption_type'], 'ebsVolumeSize': {'size': self.parameters['ebs_volume_size'], 'unit': self.parameters['ebs_volume_size_unit']}, 'ebsVolumeType': self.parameters['ebs_volume_type'], 'svmPassword': self.parameters['svm_password'], 'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs'], 'enableCompliance': self.parameters['enable_compliance'], 'enableMonitoring': self.parameters['enable_monitoring'], 'optimizedNetworkUtilization': self.parameters['optimized_network_utilization'], 'vsaMetadata': {'ontapVersion': self.parameters['ontap_version'], 'licenseType': self.parameters['license_type'], 'useLatestVersion': self.parameters['use_latest_version'], 'instanceType': self.parameters['instance_type']}}
    if self.parameters['capacity_tier'] == 'S3':
        json.update({'capacityTier': self.parameters['capacity_tier'], 'tierLevel': self.parameters['tier_level']})
    if not self.parameters['license_type'].endswith('capacity-paygo'):
        json['vsaMetadata'].update({'capacityPackageName': ''})
    if self.parameters.get('platform_serial_number') is not None:
        json['vsaMetadata'].update({'platformSerialNumber': self.parameters['platform_serial_number']})
    if self.parameters.get('provided_license') is not None:
        json['vsaMetadata'].update({'providedLicense': self.parameters['provided_license']})
    if self.parameters.get('capacity_package_name') is not None:
        json['vsaMetadata'].update({'capacityPackageName': self.parameters['capacity_package_name']})
    if self.parameters.get('writing_speed_state') is not None:
        json.update({'writingSpeedState': self.parameters['writing_speed_state'].upper()})
    if self.parameters.get('iops') is not None:
        json.update({'iops': self.parameters['iops']})
    if self.parameters.get('throughput') is not None:
        json.update({'throughput': self.parameters['throughput']})
    if self.parameters.get('cluster_key_pair_name') is not None:
        json.update({'clusterKeyPairName': self.parameters['cluster_key_pair_name']})
    if self.parameters.get('instance_tenancy') is not None:
        json.update({'instanceTenancy': self.parameters['instance_tenancy']})
    if self.parameters.get('instance_profile_name') is not None:
        json.update({'instanceProfileName': self.parameters['instance_profile_name']})
    if self.parameters.get('security_group_id') is not None:
        json.update({'securityGroupId': self.parameters['security_group_id']})
    if self.parameters.get('cloud_provider_account') is not None:
        json.update({'cloudProviderAccount': self.parameters['cloud_provider_account']})
    if self.parameters.get('backup_volumes_to_cbs') is not None:
        json.update({'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs']})
    if self.parameters.get('svm_name') is not None:
        json.update({'svmName': self.parameters['svm_name']})
    if self.parameters['data_encryption_type'] == 'AWS':
        if self.parameters.get('kms_key_id') is not None:
            json.update({'awsEncryptionParameters': {'kmsKeyId': self.parameters['kms_key_id']}})
        if self.parameters.get('kms_key_arn') is not None:
            json.update({'awsEncryptionParameters': {'kmsKeyArn': self.parameters['kms_key_arn']}})
    if self.parameters.get('aws_tag') is not None:
        tags = []
        for each_tag in self.parameters['aws_tag']:
            tag = {'tagKey': each_tag['tag_key'], 'tagValue': each_tag['tag_value']}
            tags.append(tag)
        json.update({'awsTags': tags})
    if self.parameters['is_ha'] is True:
        ha_params = dict({'mediatorAssignPublicIP': self.parameters['mediator_assign_public_ip']})
        if self.parameters.get('failover_mode'):
            ha_params['failoverMode'] = self.parameters['failover_mode']
        if self.parameters.get('node1_subnet_id'):
            ha_params['node1SubnetId'] = self.parameters['node1_subnet_id']
        if self.parameters.get('node2_subnet_id'):
            ha_params['node2SubnetId'] = self.parameters['node2_subnet_id']
        if self.parameters.get('mediator_subnet_id'):
            ha_params['mediatorSubnetId'] = self.parameters['mediator_subnet_id']
        if self.parameters.get('mediator_key_pair_name'):
            ha_params['mediatorKeyPairName'] = self.parameters['mediator_key_pair_name']
        if self.parameters.get('cluster_floating_ip'):
            ha_params['clusterFloatingIP'] = self.parameters['cluster_floating_ip']
        if self.parameters.get('data_floating_ip'):
            ha_params['dataFloatingIP'] = self.parameters['data_floating_ip']
        if self.parameters.get('data_floating_ip2'):
            ha_params['dataFloatingIP2'] = self.parameters['data_floating_ip2']
        if self.parameters.get('svm_floating_ip'):
            ha_params['svmFloatingIP'] = self.parameters['svm_floating_ip']
        if self.parameters.get('route_table_ids'):
            ha_params['routeTableIds'] = self.parameters['route_table_ids']
        if self.parameters.get('platform_serial_number_node1'):
            ha_params['platformSerialNumberNode1'] = self.parameters['platform_serial_number_node1']
        if self.parameters.get('platform_serial_number_node2'):
            ha_params['platformSerialNumberNode2'] = self.parameters['platform_serial_number_node2']
        json['haParams'] = ha_params
    else:
        json['subnetId'] = self.parameters['subnet_id']
    api_url = '%s/working-environments' % self.rest_api.api_root_path
    response, error, on_cloud_request_id = self.rest_api.post(api_url, json, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on creating cvo aws: %s, %s' % (str(error), str(response)))
    working_environment_id = response['publicId']
    wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
    err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'create', 60, 60)
    if err is not None:
        self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating CVO AWS: %s' % str(err))
    return working_environment_id