from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def find_asg_instance_refreshes(conn, module):
    """
    Args:
        conn (boto3.AutoScaling.Client): Valid Boto3 ASG client.
        module: AnsibleAWSModule object

    Returns:
        {
            "instance_refreshes": [
                    {
                        'auto_scaling_group_name': 'ansible-test-hermes-63642726-asg',
                        'instance_refresh_id': '6507a3e5-4950-4503-8978-e9f2636efc09',
                        'instances_to_update': 1,
                        'percentage_complete': 0,
                        "preferences": {
                            "instance_warmup": 60,
                            "min_healthy_percentage": 90,
                            "skip_matching": false
                        },
                        'start_time': '2021-02-04T03:39:40+00:00',
                        'status': 'Cancelled',
                        'status_reason': 'Cancelled due to user request.',
                    }
            ],
            'next_token': 'string'
        }
    """
    asg_name = module.params.get('name')
    asg_ids = module.params.get('ids')
    asg_next_token = module.params.get('next_token')
    asg_max_records = module.params.get('max_records')
    args = {}
    args['AutoScalingGroupName'] = asg_name
    if asg_ids:
        args['InstanceRefreshIds'] = asg_ids
    if asg_next_token:
        args['NextToken'] = asg_next_token
    if asg_max_records:
        args['MaxRecords'] = asg_max_records
    try:
        instance_refreshes_result = {}
        response = conn.describe_instance_refreshes(**args)
        if 'InstanceRefreshes' in response:
            instance_refreshes_dict = dict(instance_refreshes=response['InstanceRefreshes'], next_token=response.get('next_token', ''))
            instance_refreshes_result = camel_dict_to_snake_dict(instance_refreshes_dict)
        while 'NextToken' in response:
            args['NextToken'] = response['NextToken']
            response = conn.describe_instance_refreshes(**args)
            if 'InstanceRefreshes' in response:
                instance_refreshes_dict = camel_dict_to_snake_dict(dict(instance_refreshes=response['InstanceRefreshes'], next_token=response.get('next_token', '')))
                instance_refreshes_result.update(instance_refreshes_dict)
        return module.exit_json(**instance_refreshes_result)
    except (BotoCoreError, ClientError) as e:
        module.fail_json_aws(e, msg='Failed to describe InstanceRefreshes')