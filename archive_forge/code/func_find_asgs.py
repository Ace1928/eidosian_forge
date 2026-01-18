import re
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def find_asgs(conn, module, name=None, tags=None):
    """
    Args:
        conn (boto3.AutoScaling.Client): Valid Boto3 ASG client.
        name (str): Optional name of the ASG you are looking for.
        tags (dict): Optional dictionary of tags and values to search for.

    Basic Usage:
        >>> name = 'public-webapp-production'
        >>> tags = { 'env': 'production' }
        >>> conn = boto3.client('autoscaling', region_name='us-west-2')
        >>> results = find_asgs(name, conn)

    Returns:
        List
        [
            {
                "auto_scaling_group_arn": (
                    "arn:aws:autoscaling:us-west-2:123456789012:autoScalingGroup:58abc686-9783-4528-b338-3ad6f1cbbbaf:"
                    "autoScalingGroupName/public-webapp-production"
                ),
                "auto_scaling_group_name": "public-webapp-production",
                "availability_zones": ["us-west-2c", "us-west-2b", "us-west-2a"],
                "created_time": "2016-02-02T23:28:42.481000+00:00",
                "default_cooldown": 300,
                "desired_capacity": 2,
                "enabled_metrics": [],
                "health_check_grace_period": 300,
                "health_check_type": "ELB",
                "instances":
                [
                    {
                        "availability_zone": "us-west-2c",
                        "health_status": "Healthy",
                        "instance_id": "i-047a12cb",
                        "launch_configuration_name": "public-webapp-production-1",
                        "lifecycle_state": "InService",
                        "protected_from_scale_in": false
                    },
                    {
                        "availability_zone": "us-west-2a",
                        "health_status": "Healthy",
                        "instance_id": "i-7a29df2c",
                        "launch_configuration_name": "public-webapp-production-1",
                        "lifecycle_state": "InService",
                        "protected_from_scale_in": false
                    }
                ],
                "launch_config_name": "public-webapp-production-1",
                "launch_configuration_name": "public-webapp-production-1",
                "lifecycle_hooks":
                [
                    {
                        "AutoScalingGroupName": "public-webapp-production-1",
                        "DefaultResult": "ABANDON",
                        "GlobalTimeout": 172800,
                        "HeartbeatTimeout": 3600,
                        "LifecycleHookName": "instance-launch",
                        "LifecycleTransition": "autoscaling:EC2_INSTANCE_LAUNCHING"
                    },
                    {
                        "AutoScalingGroupName": "public-webapp-production-1",
                        "DefaultResult": "ABANDON",
                        "GlobalTimeout": 172800,
                        "HeartbeatTimeout": 3600,
                        "LifecycleHookName": "instance-terminate",
                        "LifecycleTransition": "autoscaling:EC2_INSTANCE_TERMINATING"
                    }
                ],
                "load_balancer_names": ["public-webapp-production-lb"],
                "max_size": 4,
                "min_size": 2,
                "new_instances_protected_from_scale_in": false,
                "placement_group": None,
                "status": None,
                "suspended_processes": [],
                "tags":
                [
                    {
                        "key": "Name",
                        "propagate_at_launch": true,
                        "resource_id": "public-webapp-production",
                        "resource_type": "auto-scaling-group",
                        "value": "public-webapp-production"
                    },
                    {
                        "key": "env",
                        "propagate_at_launch": true,
                        "resource_id": "public-webapp-production",
                        "resource_type": "auto-scaling-group",
                        "value": "production"
                    }
                ],
                "target_group_names": [],
                "target_group_arns": [],
                "termination_policies":
                [
                    "Default"
                ],
                "vpc_zone_identifier":
                [
                    "subnet-a1b1c1d1",
                    "subnet-a2b2c2d2",
                    "subnet-a3b3c3d3"
                ]
            }
        ]
    """
    try:
        asgs_paginator = conn.get_paginator('describe_auto_scaling_groups')
        asgs = asgs_paginator.paginate().build_full_result()
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to describe AutoScalingGroups')
    if not asgs:
        return asgs
    try:
        elbv2 = module.client('elbv2')
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError):
        elbv2 = None
    matched_asgs = []
    if name is not None:
        name_prog = re.compile('^' + name)
    for asg in asgs['AutoScalingGroups']:
        if name:
            matched_name = name_prog.search(asg['AutoScalingGroupName'])
        else:
            matched_name = True
        if tags:
            matched_tags = match_asg_tags(tags, asg)
        else:
            matched_tags = True
        if matched_name and matched_tags:
            asg = camel_dict_to_snake_dict(asg)
            if 'launch_configuration_name' in asg:
                asg['launch_config_name'] = asg['launch_configuration_name']
            if 'target_group_ar_ns' in asg:
                asg['target_group_arns'] = asg['target_group_ar_ns']
                del asg['target_group_ar_ns']
            if asg.get('target_group_arns'):
                if elbv2:
                    try:
                        tg_paginator = elbv2.get_paginator('describe_target_groups')
                        tg_chunk_size = 20
                        asg['target_group_names'] = []
                        tg_chunks = [asg['target_group_arns'][i:i + tg_chunk_size] for i in range(0, len(asg['target_group_arns']), tg_chunk_size)]
                        for chunk in tg_chunks:
                            tg_result = tg_paginator.paginate(TargetGroupArns=chunk).build_full_result()
                            asg['target_group_names'].extend([tg['TargetGroupName'] for tg in tg_result['TargetGroups']])
                    except is_boto3_error_code('TargetGroupNotFound'):
                        asg['target_group_names'] = []
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        module.fail_json_aws(e, msg='Failed to describe Target Groups')
            else:
                asg['target_group_names'] = []
            try:
                asg_lifecyclehooks = conn.describe_lifecycle_hooks(AutoScalingGroupName=asg['auto_scaling_group_name'])
                asg['lifecycle_hooks'] = asg_lifecyclehooks['LifecycleHooks']
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failed to fetch information about ASG lifecycle hooks')
            matched_asgs.append(asg)
    return matched_asgs