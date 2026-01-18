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
def get_restore_snapshot_options(params_dict):
    options = ['AvailabilityZones', 'BacktrackWindow', 'DBClusterIdentifier', 'DBSubnetGroupName', 'DatabaseName', 'EnableCloudwatchLogsExports', 'EnableIAMDatabaseAuthentication', 'Engine', 'EngineVersion', 'KmsKeyId', 'OptionGroupName', 'Port', 'SnapshotIdentifier', 'Tags', 'VpcSecurityGroupIds', 'DBClusterParameterGroupName', 'DeletionProtection', 'CopyTagsToSnapshot', 'Domain', 'DomainIAMRoleName']
    return dict(((k, v) for k, v in params_dict.items() if k in options and v is not None))