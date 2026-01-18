from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def get_aliases_from_distribution_id(self, distribution_id):
    try:
        distribution = self.get_distribution(id=distribution_id)
        return distribution['Distribution']['DistributionConfig']['Aliases'].get('Items', [])
    except botocore.exceptions.ClientError as e:
        self.module.fail_json_aws(e, msg='Error getting list of aliases from distribution_id')