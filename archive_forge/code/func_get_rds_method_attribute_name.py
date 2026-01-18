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
def get_rds_method_attribute_name(instance, state, creation_source, read_replica):
    method_name = None
    if state == 'absent' or state == 'terminated':
        if instance and instance['DBInstanceStatus'] not in ['deleting', 'deleted']:
            method_name = 'delete_db_instance'
    elif instance:
        method_name = 'modify_db_instance'
    elif read_replica is True:
        method_name = 'create_db_instance_read_replica'
    elif creation_source == 'snapshot':
        method_name = 'restore_db_instance_from_db_snapshot'
    elif creation_source == 's3':
        method_name = 'restore_db_instance_from_s3'
    elif creation_source == 'instance':
        method_name = 'restore_db_instance_to_point_in_time'
    else:
        method_name = 'create_db_instance'
    return method_name