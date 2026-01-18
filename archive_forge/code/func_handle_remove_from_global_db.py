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
def handle_remove_from_global_db(module, cluster):
    global_cluster_id = module.params.get('global_cluster_identifier')
    db_cluster_id = module.params.get('db_cluster_identifier')
    db_cluster_arn = cluster['DBClusterArn']
    if module.check_mode:
        return True
    try:
        client.remove_from_global_cluster(DbClusterIdentifier=db_cluster_arn, GlobalClusterIdentifier=global_cluster_id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed to remove cluster {db_cluster_id} from global DB cluster {global_cluster_id}.')
    if 'GlobalWriteForwardingStatus' in cluster:
        wait_for_cluster_status(client, module, db_cluster_id, 'db_cluster_promoting')
    if module.params.get('wait'):
        wait_for_cluster_status(client, module, db_cluster_id, 'cluster_available')
    return True