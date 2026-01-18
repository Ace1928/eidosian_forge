import json
from traceback import format_exc
from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def compare_regions(desired_secret, current_secret):
    """Compare secrets replication configuration

    Args:
        desired_secret: camel dict representation of the desired secret state.
        current_secret: secret reference as returned by the secretsmanager api.

    Returns: bool
    """
    regions_to_set_replication = []
    regions_to_remove_replication = []
    if desired_secret.replica_regions is None:
        return (regions_to_set_replication, regions_to_remove_replication)
    if desired_secret.replica_regions:
        regions_to_set_replication = desired_secret.replica_regions
    for current_secret_region in current_secret.get('ReplicationStatus', []):
        if regions_to_set_replication:
            for desired_secret_region in regions_to_set_replication:
                if current_secret_region['Region'] == desired_secret_region['region']:
                    regions_to_set_replication.remove(desired_secret_region)
                else:
                    regions_to_remove_replication.append(current_secret_region['Region'])
        else:
            regions_to_remove_replication.append(current_secret_region['Region'])
    return (regions_to_set_replication, regions_to_remove_replication)