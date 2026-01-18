from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_elasticache_clusters(client, module):
    region = module.region
    try:
        clusters = describe_cache_clusters_with_backoff(client, cluster_id=module.params.get('name'))
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg="Couldn't obtain cache cluster info")
    account_id, partition = get_aws_account_info(module)
    results = []
    for cluster in clusters:
        cluster = camel_dict_to_snake_dict(cluster)
        arn = f'arn:{partition}:elasticache:{region}:{account_id}:cluster:{cluster['cache_cluster_id']}'
        try:
            tags = get_elasticache_tags_with_backoff(client, arn)
        except is_boto3_error_code('CacheClusterNotFound'):
            continue
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg=f"Couldn't get tags for cluster {cluster['cache_cluster_id']}")
        cluster['tags'] = boto3_tag_list_to_ansible_dict(tags)
        if cluster.get('replication_group_id', None):
            try:
                replication_group = describe_replication_group_with_backoff(client, cluster['replication_group_id'])
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg="Couldn't obtain replication group info")
            if replication_group is not None:
                replication_group = camel_dict_to_snake_dict(replication_group)
                cluster['replication_group'] = replication_group
        results.append(cluster)
    return results