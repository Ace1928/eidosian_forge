from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import arg_spec_to_rds_params
from ansible_collections.amazon.aws.plugins.module_utils.rds import call_method
from ansible_collections.amazon.aws.plugins.module_utils.rds import compare_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_final_identifier
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_rds_method_attribute
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import update_iam_roles
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_current_attributes_with_inconsistent_keys(instance):
    options = {}
    if instance.get('PendingModifiedValues', {}).get('PendingCloudwatchLogsExports', {}).get('LogTypesToEnable', []):
        current_enabled = instance['PendingModifiedValues']['PendingCloudwatchLogsExports']['LogTypesToEnable']
        current_disabled = instance['PendingModifiedValues']['PendingCloudwatchLogsExports']['LogTypesToDisable']
        options['CloudwatchLogsExportConfiguration'] = {'LogTypesToEnable': current_enabled, 'LogTypesToDisable': current_disabled}
    else:
        options['CloudwatchLogsExportConfiguration'] = {'LogTypesToEnable': instance.get('EnabledCloudwatchLogsExports', []), 'LogTypesToDisable': []}
    if instance.get('PendingModifiedValues', {}).get('Port'):
        options['DBPortNumber'] = instance['PendingModifiedValues']['Port']
    else:
        options['DBPortNumber'] = instance['Endpoint']['Port']
    if instance.get('PendingModifiedValues', {}).get('DBSubnetGroupName'):
        options['DBSubnetGroupName'] = instance['PendingModifiedValues']['DBSubnetGroupName']
    else:
        options['DBSubnetGroupName'] = instance['DBSubnetGroup']['DBSubnetGroupName']
    if instance.get('PendingModifiedValues', {}).get('ProcessorFeatures'):
        options['ProcessorFeatures'] = instance['PendingModifiedValues']['ProcessorFeatures']
    else:
        options['ProcessorFeatures'] = instance.get('ProcessorFeatures', {})
    options['OptionGroupName'] = [g['OptionGroupName'] for g in instance['OptionGroupMemberships']]
    options['DBSecurityGroups'] = [sg['DBSecurityGroupName'] for sg in instance['DBSecurityGroups'] if sg['Status'] in ['adding', 'active']]
    options['VpcSecurityGroupIds'] = [sg['VpcSecurityGroupId'] for sg in instance['VpcSecurityGroups'] if sg['Status'] in ['adding', 'active']]
    options['DBParameterGroupName'] = [parameter_group['DBParameterGroupName'] for parameter_group in instance['DBParameterGroups']]
    options['EnableIAMDatabaseAuthentication'] = instance['IAMDatabaseAuthenticationEnabled']
    options['EnablePerformanceInsights'] = instance.get('PerformanceInsightsEnabled', False)
    options['NewDBInstanceIdentifier'] = instance['DBInstanceIdentifier']
    options['AllowMajorVersionUpgrade'] = None
    options['MasterUserPassword'] = None
    return options