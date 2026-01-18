from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def get_etag_from_distribution_id(self, distribution_id, streaming):
    distribution = {}
    if not streaming:
        distribution = self.get_distribution(id=distribution_id)
    else:
        distribution = self.get_streaming_distribution(id=distribution_id)
    return distribution['ETag']