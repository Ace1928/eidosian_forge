import copy
import hashlib
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import botocore
from ray.autoscaler._private.aws.utils import client_cache, resource_cache
from ray.autoscaler.tags import NODE_KIND_HEAD, TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_KIND
@staticmethod
def resolve_policy_arns(config: Dict[str, Any], iam: Any, default_policy_arns: List[str]) -> List[str]:
    """Attach necessary AWS policies for CloudWatch related operations.

        Args:
            config: provider section of cluster config file.
            iam: AWS iam resource.
            default_policy_arns: List of default ray AWS policies.

        Returns:
            list of policy arns including additional policies for CloudWatch
                related operations if cloudwatch agent config is specifed in
                cluster config file.
        """
    cwa_cfg_exists = CloudwatchHelper.cloudwatch_config_exists(config, CloudwatchConfigType.AGENT.value)
    if cwa_cfg_exists:
        cloudwatch_managed_policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['ssm:SendCommand', 'ssm:ListCommandInvocations', 'iam:PassRole'], 'Resource': '*'}]}
        iam_client = iam.meta.client
        iam_client.create_policy(PolicyName='CloudwatchManagedPolicies', PolicyDocument=json.dumps(cloudwatch_managed_policy))
        sts_client = client_cache('sts', config['region'])
        account_id = sts_client.get_caller_identity().get('Account')
        managed_policy_arn = 'arn:aws:iam::{}:policy/CloudwatchManagedPolicies'.format(account_id)
        policy_waiter = iam_client.get_waiter('policy_exists')
        policy_waiter.wait(PolicyArn=managed_policy_arn, WaiterConfig={'Delay': 2, 'MaxAttempts': 200})
        new_policy_arns = copy.copy(default_policy_arns)
        new_policy_arns.extend(['arn:aws:iam::aws:policy/CloudWatchAgentAdminPolicy', 'arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore', managed_policy_arn])
        return new_policy_arns
    else:
        return default_policy_arns