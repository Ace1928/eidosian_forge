from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def create_cvo_gcp(self):
    if self.parameters.get('workspace_id') is None:
        response, msg = self.na_helper.get_tenant(self.rest_api, self.headers)
        if response is None:
            self.module.fail_json(msg)
        self.parameters['workspace_id'] = response
    if self.parameters.get('nss_account') is None:
        if self.parameters.get('platform_serial_number') is not None:
            if not self.parameters['platform_serial_number'].startswith('Eval-'):
                if self.parameters['license_type'] == 'gcp-cot-premium-byol' or self.parameters['license_type'] == 'gcp-ha-cot-premium-byol':
                    response, msg = self.na_helper.get_nss(self.rest_api, self.headers)
                    if response is None:
                        self.module.fail_json(msg)
                    self.parameters['nss_account'] = response
    if self.parameters['is_ha'] is True and self.parameters['license_type'] == 'capacity-paygo':
        self.parameters['license_type'] == 'ha-capacity-paygo'
    json = {'name': self.parameters['name'], 'region': self.parameters['zone'], 'tenantId': self.parameters['workspace_id'], 'vpcId': self.parameters['vpc_id'], 'gcpServiceAccount': self.parameters['gcp_service_account'], 'gcpVolumeSize': {'size': self.parameters['gcp_volume_size'], 'unit': self.parameters['gcp_volume_size_unit']}, 'gcpVolumeType': self.parameters['gcp_volume_type'], 'svmPassword': self.parameters['svm_password'], 'backupVolumesToCbs': self.parameters['backup_volumes_to_cbs'], 'enableCompliance': self.parameters['enable_compliance'], 'vsaMetadata': {'ontapVersion': self.parameters['ontap_version'], 'licenseType': self.parameters['license_type'], 'useLatestVersion': self.parameters['use_latest_version'], 'instanceType': self.parameters['instance_type']}}
    if self.parameters['is_ha'] is False:
        if self.parameters.get('writing_speed_state') is None:
            self.parameters['writing_speed_state'] = 'NORMAL'
        json.update({'writingSpeedState': self.parameters['writing_speed_state'].upper()})
    if self.parameters.get('data_encryption_type') is not None and self.parameters['data_encryption_type'] == 'GCP':
        json.update({'dataEncryptionType': self.parameters['data_encryption_type']})
        if self.parameters.get('gcp_encryption_parameters') is not None:
            json.update({'gcpEncryptionParameters': {'key': self.parameters['gcp_encryption_parameters']}})
    if self.parameters.get('provided_license') is not None:
        json['vsaMetadata'].update({'providedLicense': self.parameters['provided_license']})
    if not self.parameters['license_type'].endswith('capacity-paygo'):
        json['vsaMetadata'].update({'capacityPackageName': ''})
    if self.parameters.get('capacity_package_name') is not None:
        json['vsaMetadata'].update({'capacityPackageName': self.parameters['capacity_package_name']})
    if self.parameters.get('project_id'):
        json.update({'project': self.parameters['project_id']})
    if self.parameters.get('nss_account'):
        json.update({'nssAccount': self.parameters['nss_account']})
    if self.parameters.get('subnet_id'):
        json.update({'subnetId': self.parameters['subnet_id']})
    if self.parameters.get('subnet_path'):
        json.update({'subnetPath': self.parameters['subnet_path']})
    if self.parameters.get('platform_serial_number') is not None:
        json.update({'serialNumber': self.parameters['platform_serial_number']})
    if self.parameters.get('capacity_tier') is not None and self.parameters['capacity_tier'] == 'cloudStorage':
        json.update({'capacityTier': self.parameters['capacity_tier'], 'tierLevel': self.parameters['tier_level']})
    if self.parameters.get('svm_name') is not None:
        json.update({'svmName': self.parameters['svm_name']})
    if self.parameters.get('gcp_labels') is not None:
        labels = []
        for each_label in self.parameters['gcp_labels']:
            label = {'labelKey': each_label['label_key'], 'labelValue': each_label['label_value']}
            labels.append(label)
        json.update({'gcpLabels': labels})
    if self.parameters.get('firewall_rule'):
        json.update({'firewallRule': self.parameters['firewall_rule']})
    if self.parameters['is_ha'] is True:
        ha_params = dict()
        if self.parameters.get('network_project_id') is not None:
            network_project_id = self.parameters.get('network_project_id')
        else:
            network_project_id = self.parameters['project_id']
        if not self.has_self_link(self.parameters['subnet_id']):
            json.update({'subnetId': 'projects/%s/regions/%s/subnetworks/%s' % (network_project_id, self.parameters['zone'][:-2], self.parameters['subnet_id'])})
        if self.parameters.get('platform_serial_number_node1'):
            ha_params['platformSerialNumberNode1'] = self.parameters['platform_serial_number_node1']
        if self.parameters.get('platform_serial_number_node2'):
            ha_params['platformSerialNumberNode2'] = self.parameters['platform_serial_number_node2']
        if self.parameters.get('node1_zone'):
            ha_params['node1Zone'] = self.parameters['node1_zone']
        if self.parameters.get('node2_zone'):
            ha_params['node2Zone'] = self.parameters['node2_zone']
        if self.parameters.get('mediator_zone'):
            ha_params['mediatorZone'] = self.parameters['mediator_zone']
        if self.parameters.get('vpc0_node_and_data_connectivity'):
            if self.has_self_link(self.parameters['vpc0_node_and_data_connectivity']):
                ha_params['vpc0NodeAndDataConnectivity'] = self.parameters['vpc0_node_and_data_connectivity']
            else:
                ha_params['vpc0NodeAndDataConnectivity'] = GOOGLE_API_URL + '/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc0_node_and_data_connectivity'])
        if self.parameters.get('vpc1_cluster_connectivity'):
            if self.has_self_link(self.parameters['vpc1_cluster_connectivity']):
                ha_params['vpc1ClusterConnectivity'] = self.parameters['vpc1_cluster_connectivity']
            else:
                ha_params['vpc1ClusterConnectivity'] = GOOGLE_API_URL + '/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc1_cluster_connectivity'])
        if self.parameters.get('vpc2_ha_connectivity'):
            if self.has_self_link(self.parameters['vpc2_ha_connectivity']):
                ha_params['vpc2HAConnectivity'] = self.parameters['vpc2_ha_connectivity']
            else:
                ha_params['vpc2HAConnectivity'] = 'https://www.googleapis.com/compute/v1/projects/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc2_ha_connectivity'])
        if self.parameters.get('vpc3_data_replication'):
            if self.has_self_link(self.parameters['vpc3_data_replication']):
                ha_params['vpc3DataReplication'] = self.parameters['vpc3_data_replication']
            else:
                ha_params['vpc3DataReplication'] = GOOGLE_API_URL + '/{0}/global/networks/{1}'.format(network_project_id, self.parameters['vpc3_data_replication'])
        if self.parameters.get('subnet0_node_and_data_connectivity'):
            if self.has_self_link(self.parameters['subnet0_node_and_data_connectivity']):
                ha_params['subnet0NodeAndDataConnectivity'] = self.parameters['subnet0_node_and_data_connectivity']
            else:
                ha_params['subnet0NodeAndDataConnectivity'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet0_node_and_data_connectivity'])
        if self.parameters.get('subnet1_cluster_connectivity'):
            if self.has_self_link(self.parameters['subnet1_cluster_connectivity']):
                ha_params['subnet1ClusterConnectivity'] = self.parameters['subnet1_cluster_connectivity']
            else:
                ha_params['subnet1ClusterConnectivity'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet1_cluster_connectivity'])
        if self.parameters.get('subnet2_ha_connectivity'):
            if self.has_self_link(self.parameters['subnet2_ha_connectivity']):
                ha_params['subnet2HAConnectivity'] = self.parameters['subnet2_ha_connectivity']
            else:
                ha_params['subnet2HAConnectivity'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet2_ha_connectivity'])
        if self.parameters.get('subnet3_data_replication'):
            if self.has_self_link(self.parameters['subnet3_data_replication']):
                ha_params['subnet3DataReplication'] = self.parameters['subnet3_data_replication']
            else:
                ha_params['subnet3DataReplication'] = GOOGLE_API_URL + '/{0}/regions/{1}/subnetworks/{2}'.format(network_project_id, self.parameters['zone'][:-2], self.parameters['subnet3_data_replication'])
        if self.parameters.get('vpc0_firewall_rule_name'):
            ha_params['vpc0FirewallRuleName'] = self.parameters['vpc0_firewall_ruleName']
        if self.parameters.get('vpc1_firewall_rule_name'):
            ha_params['vpc1FirewallRuleName'] = self.parameters['vpc1_firewall_rule_name']
        if self.parameters.get('vpc2_firewall_rule_name'):
            ha_params['vpc2FirewallRuleName'] = self.parameters['vpc2_firewall_rule_name']
        if self.parameters.get('vpc3_firewall_rule_name'):
            ha_params['vpc3FirewallRuleName'] = self.parameters['vpc3_firewall_rule_name']
        json['haParams'] = ha_params
    api_url = '%s/working-environments' % self.rest_api.api_root_path
    response, error, on_cloud_request_id = self.rest_api.post(api_url, json, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on creating cvo gcp: %s, %s' % (str(error), str(response)))
    working_environment_id = response['publicId']
    wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
    err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'CVO', 'create', 60, 60)
    if err is not None:
        self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating CVO GCP: %s' % str(err))
    return working_environment_id