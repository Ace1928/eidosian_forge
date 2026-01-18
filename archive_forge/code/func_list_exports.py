from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.exponential_backoff()
def list_exports(cloudformation_client):
    """Get Exports Names and Values and return in dictionary"""
    list_exports_paginator = cloudformation_client.get_paginator('list_exports')
    exports = list_exports_paginator.paginate().build_full_result()['Exports']
    export_items = dict()
    for item in exports:
        export_items[item['Name']] = item['Value']
    return export_items