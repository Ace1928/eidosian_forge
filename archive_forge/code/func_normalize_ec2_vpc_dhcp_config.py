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
def normalize_ec2_vpc_dhcp_config(option_config):
    """
    The boto2 module returned a config dict, but boto3 returns a list of dicts
    Make the data we return look like the old way, so we don't break users.
    This is also much more user-friendly.
    boto3:
        'DhcpConfigurations': [
            {'Key': 'domain-name', 'Values': [{'Value': 'us-west-2.compute.internal'}]},
            {'Key': 'domain-name-servers', 'Values': [{'Value': 'AmazonProvidedDNS'}]},
            {'Key': 'netbios-name-servers', 'Values': [{'Value': '1.2.3.4'}, {'Value': '5.6.7.8'}]},
            {'Key': 'netbios-node-type', 'Values': [1]},
            {'Key': 'ntp-servers', 'Values': [{'Value': '1.2.3.4'}, {'Value': '5.6.7.8'}]}
        ],
    The module historically returned:
        "new_options": {
            "domain-name": "ec2.internal",
            "domain-name-servers": ["AmazonProvidedDNS"],
            "netbios-name-servers": ["10.0.0.1", "10.0.1.1"],
            "netbios-node-type": "1",
            "ntp-servers": ["10.0.0.2", "10.0.1.2"]
        },
    """
    config_data = {}
    if len(option_config) == 0:
        return config_data
    for config_item in option_config:
        if config_item['Key'] == 'netbios-node-type':
            if isinstance(config_item['Values'], integer_types):
                config_data['netbios-node-type'] = str(config_item['Values'])
            elif isinstance(config_item['Values'], list):
                config_data['netbios-node-type'] = str(config_item['Values'][0]['Value'])
        for option in ['domain-name', 'domain-name-servers', 'ntp-servers', 'netbios-name-servers']:
            if config_item['Key'] == option:
                config_data[option] = [val['Value'] for val in config_item['Values']]
    return config_data