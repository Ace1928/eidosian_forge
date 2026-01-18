from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def create_snapmirror(self):
    snapmirror_build_data = {}
    replication_request = {}
    replication_volume = {}
    source_we_info, dest_we_info, err = self.na_helper.get_working_environment_detail_for_snapmirror(self.rest_api, self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg=err)
    if self.parameters.get('capacity_tier') is not None:
        if self.parameters['capacity_tier'] == 'NONE':
            self.parameters.pop('capacity_tier')
    elif dest_we_info.get('cloudProviderName'):
        self.parameters['capacity_tier'] = PROVIDER_TO_CAPACITY_TIER[dest_we_info['cloudProviderName'].lower()]
    interclusterlifs_info = self.get_interclusterlifs(source_we_info['publicId'], dest_we_info['publicId'])
    if source_we_info['workingEnvironmentType'] != 'ON_PREM':
        source_volumes = self.get_volumes(source_we_info, self.parameters['source_volume_name'])
    else:
        source_volumes = self.get_volumes_on_prem(source_we_info, self.parameters['source_volume_name'])
    if len(source_volumes) == 0:
        self.module.fail_json(changed=False, msg='source volume not found')
    vol_found = False
    vol_dest_quote = {}
    source_volume_resp = {}
    for vol in source_volumes:
        if vol['name'] == self.parameters['source_volume_name']:
            vol_found = True
            vol_dest_quote = vol
            source_volume_resp = vol
            if self.parameters.get('source_svm_name') is not None and vol['svmName'] != self.parameters['source_svm_name']:
                vol_found = False
            if vol_found:
                break
    if not vol_found:
        self.module.fail_json(changed=False, msg='source volume not found')
    if self.parameters.get('source_svm_name') is None:
        self.parameters['source_svm_name'] = source_volume_resp['svmName']
    if self.parameters.get('destination_svm_name') is None:
        if dest_we_info.get('svmName') is not None:
            self.parameters['destination_svm_name'] = dest_we_info['svmName']
        else:
            self.parameters['destination_working_environment_name'] = dest_we_info['name']
            dest_working_env_detail, err = self.na_helper.get_working_environment_details_by_name(self.rest_api, self.headers, self.parameters['destination_working_environment_name'])
            if err:
                self.module.fail_json(changed=False, msg='Error getting destination info %s: %s.' % (err, dest_working_env_detail))
            self.parameters['destination_svm_name'] = dest_working_env_detail['svmName']
    if dest_we_info.get('workingEnvironmentType') and dest_we_info['workingEnvironmentType'] != 'ON_PREM' and (not dest_we_info['publicId'].startswith('fs-')):
        quote = self.build_quote_request(source_we_info, dest_we_info, vol_dest_quote)
        quote_response = self.quote_volume(quote)
        replication_volume['numOfDisksApprovedToAdd'] = int(quote_response['numOfDisks'])
        if 'iops' in quote:
            replication_volume['iops'] = quote['iops']
        if 'throughput' in quote:
            replication_volume['throughput'] = quote['throughput']
        if self.parameters.get('destination_aggregate_name') is not None:
            replication_volume['advancedMode'] = True
        else:
            replication_volume['advancedMode'] = False
            replication_volume['destinationAggregateName'] = quote_response['aggregateName']
    if self.parameters.get('provider_volume_type') is None:
        replication_volume['destinationProviderVolumeType'] = source_volume_resp['providerVolumeType']
    if self.parameters.get('capacity_tier') is not None:
        replication_volume['destinationCapacityTier'] = self.parameters['capacity_tier']
    replication_request['sourceWorkingEnvironmentId'] = source_we_info['publicId']
    if dest_we_info['publicId'].startswith('fs-'):
        replication_request['destinationFsxId'] = dest_we_info['publicId']
    else:
        replication_request['destinationWorkingEnvironmentId'] = dest_we_info['publicId']
    replication_volume['sourceVolumeName'] = self.parameters['source_volume_name']
    replication_volume['destinationVolumeName'] = self.parameters['destination_volume_name']
    replication_request['policyName'] = self.parameters['policy']
    replication_request['scheduleName'] = self.parameters['schedule']
    replication_request['maxTransferRate'] = self.parameters['max_transfer_rate']
    replication_volume['sourceSvmName'] = source_volume_resp['svmName']
    replication_volume['destinationSvmName'] = self.parameters['destination_svm_name']
    replication_request['sourceInterclusterLifIps'] = [interclusterlifs_info['interClusterLifs'][0]['address']]
    replication_request['destinationInterclusterLifIps'] = [interclusterlifs_info['peerInterClusterLifs'][0]['address']]
    snapmirror_build_data['replicationRequest'] = replication_request
    snapmirror_build_data['replicationVolume'] = replication_volume
    if dest_we_info['publicId'].startswith('fs-'):
        api = '/occm/api/replication/fsx'
    elif dest_we_info['workingEnvironmentType'] != 'ON_PREM':
        api = '/occm/api/replication/vsa'
    else:
        api = '/occm/api/replication/onprem'
    response, err, on_cloud_request_id = self.rest_api.send_request('POST', api, None, snapmirror_build_data, header=self.headers)
    if err is not None:
        self.module.fail_json(changed=False, msg='Error creating snapmirror relationship %s: %s.' % (err, response))
    wait_on_completion_api_url = '/occm/api/audit/activeTask/%s' % str(on_cloud_request_id)
    err = self.rest_api.wait_on_completion(wait_on_completion_api_url, 'snapmirror', 'create', 20, 5)
    if err is not None:
        self.module.fail_json(changed=False, msg=err)