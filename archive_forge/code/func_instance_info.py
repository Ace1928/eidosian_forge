from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def instance_info(conn, instance_name, filters):
    params = {}
    if instance_name:
        params['DBInstanceIdentifier'] = instance_name
    if filters:
        params['Filters'] = ansible_dict_to_boto3_filter_list(filters)
    try:
        results = _describe_db_instances(conn, **params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise RdsInstanceInfoFailure(e, "Couldn't get instance information")
    for instance in results:
        instance['Tags'] = get_instance_tags(conn, arn=instance['DBInstanceArn'])
    return {'changed': False, 'instances': [camel_dict_to_snake_dict(instance, ignore_list=['Tags']) for instance in results]}