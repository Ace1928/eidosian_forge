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
def get_options_with_changing_values(client, module, parameters):
    instance_id = module.params['db_instance_identifier']
    purge_cloudwatch_logs = module.params['purge_cloudwatch_logs_exports']
    force_update_password = module.params['force_update_password']
    port = module.params['port']
    apply_immediately = parameters.pop('ApplyImmediately', None)
    cloudwatch_logs_enabled = module.params['enable_cloudwatch_logs_exports']
    purge_security_groups = module.params['purge_security_groups']
    ca_certificate_identifier = module.params['ca_certificate_identifier']
    if ca_certificate_identifier:
        parameters['CACertificateIdentifier'] = ca_certificate_identifier
    if port:
        parameters['DBPortNumber'] = port
    if not force_update_password:
        parameters.pop('MasterUserPassword', None)
    if cloudwatch_logs_enabled:
        parameters['CloudwatchLogsExportConfiguration'] = cloudwatch_logs_enabled
    if not module.params['storage_type']:
        parameters.pop('Iops', None)
    instance = get_instance(client, module, instance_id)
    updated_parameters = get_changing_options_with_inconsistent_keys(parameters, instance, purge_cloudwatch_logs, purge_security_groups)
    updated_parameters.update(get_changing_options_with_consistent_keys(parameters, instance))
    parameters = updated_parameters
    if instance.get('StorageType') == 'io1':
        current_iops = instance.get('PendingModifiedValues', {}).get('Iops', instance['Iops'])
        current_allocated_storage = instance.get('PendingModifiedValues', {}).get('AllocatedStorage', instance['AllocatedStorage'])
        new_iops = module.params.get('iops')
        new_allocated_storage = module.params.get('allocated_storage')
        if current_iops != new_iops or current_allocated_storage != new_allocated_storage:
            parameters['AllocatedStorage'] = new_allocated_storage
            parameters['Iops'] = new_iops
    if instance.get('StorageType') == 'gp3':
        GP3_THROUGHPUT = True
        current_storage_throughput = instance.get('PendingModifiedValues', {}).get('StorageThroughput', instance['StorageThroughput'])
        new_storage_throughput = module.params.get('storage_throughput') or current_storage_throughput
        if new_storage_throughput != current_storage_throughput:
            parameters['StorageThroughput'] = new_storage_throughput
        current_iops = instance.get('PendingModifiedValues', {}).get('Iops', instance['Iops'])
        new_iops = module.params.get('iops') or current_iops
        new_allocated_storage = module.params.get('allocated_storage')
        current_allocated_storage = instance.get('PendingModifiedValues', {}).get('AllocatedStorage', instance['AllocatedStorage'])
        if new_allocated_storage:
            if current_allocated_storage != new_allocated_storage:
                parameters['AllocatedStorage'] = new_allocated_storage
            if new_allocated_storage >= 400:
                if new_iops < 12000:
                    module.fail_json(msg='IOPS must be at least 12000 when the allocated storage is larger than or equal to 400 GB.')
                if new_storage_throughput < 500 and GP3_THROUGHPUT:
                    module.fail_json(msg='Storage Throughput must be at least 500 when the allocated storage is larger than or equal to 400 GB.')
                if current_iops != new_iops:
                    parameters['Iops'] = new_iops
                    parameters['AllocatedStorage'] = new_allocated_storage
    if parameters.get('NewDBInstanceIdentifier') and instance.get('PendingModifiedValues', {}).get('DBInstanceIdentifier'):
        if parameters['NewDBInstanceIdentifier'] == instance['PendingModifiedValues']['DBInstanceIdentifier'] and (not apply_immediately):
            parameters.pop('NewDBInstanceIdentifier')
    if parameters:
        parameters['DBInstanceIdentifier'] = instance_id
        if apply_immediately is not None:
            parameters['ApplyImmediately'] = apply_immediately
    return parameters