from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_modify_cvo_params(self, rest_api, headers, desired, provider):
    modified = []
    if desired['update_svm_password']:
        modified = ['svm_password']
    properties = ['status', 'ontapClusterProperties.fields(upgradeVersions)']
    if provider == 'aws':
        properties.append('awsProperties')
    else:
        properties.append('providerProperties')
    we, err = self.get_working_environment_property(rest_api, headers, properties)
    if err is not None:
        return (None, err)
    if we['status'] is None or we['status']['status'] != 'ON':
        return (None, 'Error: get_modify_cvo_params working environment %s status is not ON. Operation cannot be performed.' % we['publicId'])
    tier_level = None
    if we['ontapClusterProperties']['capacityTierInfo'] is not None:
        tier_level = we['ontapClusterProperties']['capacityTierInfo']['tierLevel']
    if tier_level is not None and tier_level != desired['tier_level']:
        if provider == 'azure':
            if desired['capacity_tier'] == 'Blob':
                modified.append('tier_level')
        elif provider == 'aws':
            if desired['capacity_tier'] == 'S3':
                modified.append('tier_level')
        elif provider == 'gcp':
            if desired['capacity_tier'] == 'cloudStorage':
                modified.append('tier_level')
    if 'svm_name' in desired and we['svmName'] != desired['svm_name']:
        modified.append('svm_name')
    if 'writing_speed_state' in desired:
        if we['ontapClusterProperties']['writingSpeedState'] != desired['writing_speed_state'].upper():
            modified.append('writing_speed_state')
    if provider == 'aws':
        current_instance_type = we['awsProperties']['instances'][0]['instanceType']
        region = we['awsProperties']['regionName']
    else:
        current_instance_type = we['providerProperties']['instanceType']
        region = we['providerProperties']['regionName']
    if current_instance_type != desired['instance_type']:
        modified.append('instance_type')
    current_license_type, error = self.get_license_type(rest_api, headers, provider, region, current_instance_type, we['ontapClusterProperties']['ontapVersion'], we['ontapClusterProperties']['licenseType']['name'])
    if err is not None:
        return (None, error)
    if current_license_type != desired['license_type']:
        modified.append('license_type')
    if desired['upgrade_ontap_version'] is True:
        if desired['use_latest_version'] or desired['ontap_version'] == 'latest':
            return (None, 'Error: To upgrade ONTAP image, the ontap_version must be a specific version')
        current_version = 'ONTAP-' + we['ontapClusterProperties']['ontapVersion']
        if not desired['ontap_version'].startswith(current_version):
            if we['ontapClusterProperties']['upgradeVersions'] is not None:
                available_versions = []
                for image_info in we['ontapClusterProperties']['upgradeVersions']:
                    available_versions.append(image_info['imageVersion'])
                    if desired['ontap_version'].startswith(image_info['imageVersion']):
                        modified.append('ontap_version')
                        break
                else:
                    return (None, 'Error: No ONTAP image available for version %s. Available versions: %s' % (desired['ontap_version'], available_versions))
    tag_name = {'aws': 'aws_tag', 'azure': 'azure_tag', 'gcp': 'gcp_labels'}
    need_change, error = self.is_cvo_tags_changed(rest_api, headers, desired, tag_name[provider])
    if error is not None:
        return (None, error)
    if need_change:
        modified.append(tag_name[provider])
    for key, value in desired.items():
        if key == 'project_id' and we['providerProperties']['projectName'] != value:
            modified.append('project_id')
        if key == 'zone' and we['providerProperties']['zoneName'][0] != value:
            modified.append('zone')
        if key == 'cidr' and we['providerProperties']['vnetCidr'] != value:
            modified.append('cidr')
        if key == 'location' and we['providerProperties']['regionName'] != value:
            modified.append('location')
        if key == 'availability_zone' and we['providerProperties']['availabilityZone'] != value:
            modified.append('availability_zone')
    if modified:
        self.changed = True
    return (modified, None)