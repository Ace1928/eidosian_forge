import re
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.common.dict_transformations import _camel_to_snake  # pylint: disable=unused-import
from ansible.module_utils.common.dict_transformations import _snake_to_camel  # pylint: disable=unused-import
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict  # pylint: disable=unused-import
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict  # pylint: disable=unused-import
from ansible.module_utils.six import integer_types
from ansible.module_utils.six import string_types
from .arn import is_outpost_arn as is_outposts_arn  # pylint: disable=unused-import
from .botocore import HAS_BOTO3  # pylint: disable=unused-import
from .botocore import boto3_conn  # pylint: disable=unused-import
from .botocore import boto3_inventory_conn  # pylint: disable=unused-import
from .botocore import boto_exception  # pylint: disable=unused-import
from .botocore import get_aws_connection_info  # pylint: disable=unused-import
from .botocore import get_aws_region  # pylint: disable=unused-import
from .botocore import paginated_query_with_retries
from .exceptions import AnsibleAWSError  # pylint: disable=unused-import
from .modules import _aws_common_argument_spec as aws_common_argument_spec  # pylint: disable=unused-import
from .modules import aws_argument_spec as ec2_argument_spec  # pylint: disable=unused-import
from .policy import _py3cmp as py3cmp  # pylint: disable=unused-import
from .policy import compare_policies  # pylint: disable=unused-import
from .policy import sort_json_policy_dict  # pylint: disable=unused-import
from .retries import AWSRetry  # pylint: disable=unused-import
from .tagging import ansible_dict_to_boto3_tag_list  # pylint: disable=unused-import
from .tagging import boto3_tag_list_to_ansible_dict  # pylint: disable=unused-import
from .tagging import compare_aws_tags  # pylint: disable=unused-import
from .transformation import ansible_dict_to_boto3_filter_list  # pylint: disable=unused-import
from .transformation import map_complex_type  # pylint: disable=unused-import
def remove_ec2_tags(client, module, resource_id, tags_to_unset, retry_codes=None):
    """
    Removes Tags from an EC2 resource.

    :param client: an EC2 boto3 client
    :param module: an AnsibleAWSModule object
    :param resource_id: the identifier for the resource
    :param tags_to_unset: a list of tag keys to removes
    :param retry_codes: additional boto3 error codes to trigger retries
    """
    if not tags_to_unset:
        return False
    if module.check_mode:
        return True
    if not retry_codes:
        retry_codes = []
    tags_to_remove = [dict(Key=tagkey) for tagkey in tags_to_unset]
    try:
        AWSRetry.jittered_backoff(retries=10, catch_extra_error_codes=retry_codes)(client.delete_tags)(Resources=[resource_id], Tags=tags_to_remove)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Unable to delete tags {tags_to_unset} from {resource_id}')
    return True