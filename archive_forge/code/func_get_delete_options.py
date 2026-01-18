from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import arg_spec_to_rds_params
from ansible_collections.amazon.aws.plugins.module_utils.rds import call_method
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import wait_for_cluster_status
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def get_delete_options(params_dict):
    options = ['DBClusterIdentifier', 'FinalSnapshotIdentifier', 'SkipFinalSnapshot']
    return dict(((k, params_dict[k]) for k in options if params_dict[k] is not None))