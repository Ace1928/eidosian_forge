from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def list_gateways(self):
    try:
        paginator = self.client.get_paginator('list_gateways')
        response = paginator.paginate(PaginationConfig={'PageSize': 100}).build_full_result()
        gateways = []
        for gw in response['Gateways']:
            gateways.append(camel_dict_to_snake_dict(gw))
        return gateways
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg="Couldn't list storage gateways")