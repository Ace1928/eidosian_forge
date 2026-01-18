from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_load_balancers(eg_launchspec, load_balancers, target_group_arns):
    if load_balancers is not None or target_group_arns is not None:
        eg_load_balancers_config = spotinst.aws_elastigroup.LoadBalancersConfig()
        eg_total_lbs = []
        if load_balancers is not None:
            for elb_name in load_balancers:
                eg_elb = spotinst.aws_elastigroup.LoadBalancer()
                if elb_name is not None:
                    eg_elb.name = elb_name
                    eg_elb.type = 'CLASSIC'
                    eg_total_lbs.append(eg_elb)
        if target_group_arns is not None:
            for target_arn in target_group_arns:
                eg_elb = spotinst.aws_elastigroup.LoadBalancer()
                if target_arn is not None:
                    eg_elb.arn = target_arn
                    eg_elb.type = 'TARGET_GROUP'
                    eg_total_lbs.append(eg_elb)
        if len(eg_total_lbs) > 0:
            eg_load_balancers_config.load_balancers = eg_total_lbs
            eg_launchspec.load_balancers_config = eg_load_balancers_config