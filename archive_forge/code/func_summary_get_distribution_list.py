from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def summary_get_distribution_list(self, streaming=False):
    try:
        list_name = 'streaming_distributions' if streaming else 'distributions'
        key_list = ['Id', 'ARN', 'Status', 'LastModifiedTime', 'DomainName', 'Comment', 'PriceClass', 'Enabled']
        distribution_list = {list_name: []}
        distributions = self.list_streaming_distributions(keyed=False) if streaming else self.list_distributions(keyed=False)
        for dist in distributions:
            temp_distribution = {k: dist[k] for k in key_list}
            temp_distribution['Aliases'] = list(dist['Aliases'].get('Items', []))
            temp_distribution['ETag'] = self.get_etag_from_distribution_id(dist['Id'], streaming)
            if not streaming:
                temp_distribution['WebACLId'] = dist['WebACLId']
                invalidation_ids = self.get_list_of_invalidation_ids_from_distribution_id(dist['Id'])
                if invalidation_ids:
                    temp_distribution['Invalidations'] = invalidation_ids
            resource_tags = self.list_resource_tags(dist['ARN'])
            temp_distribution['Tags'] = boto3_tag_list_to_ansible_dict(resource_tags['Tags'].get('Items', []))
            distribution_list[list_name].append(temp_distribution)
        return distribution_list
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Error generating summary of distributions')