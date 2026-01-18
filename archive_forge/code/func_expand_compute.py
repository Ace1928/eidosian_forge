from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_compute(eg, module, is_update, do_not_update):
    elastic_ips = module.params['elastic_ips']
    on_demand_instance_type = module.params.get('on_demand_instance_type')
    spot_instance_types = module.params['spot_instance_types']
    ebs_volume_pool = module.params['ebs_volume_pool']
    availability_zones_list = module.params['availability_zones']
    product = module.params.get('product')
    eg_compute = spotinst.aws_elastigroup.Compute()
    if product is not None:
        if is_update is not True:
            eg_compute.product = product
    if elastic_ips is not None:
        eg_compute.elastic_ips = elastic_ips
    if on_demand_instance_type or spot_instance_types is not None:
        eg_instance_types = spotinst.aws_elastigroup.InstanceTypes()
        if on_demand_instance_type is not None:
            eg_instance_types.spot = spot_instance_types
        if spot_instance_types is not None:
            eg_instance_types.ondemand = on_demand_instance_type
        if eg_instance_types.spot is not None or eg_instance_types.ondemand is not None:
            eg_compute.instance_types = eg_instance_types
    expand_ebs_volume_pool(eg_compute, ebs_volume_pool)
    eg_compute.availability_zones = expand_list(availability_zones_list, az_fields, 'AvailabilityZone')
    expand_launch_spec(eg_compute, module, is_update, do_not_update)
    eg.compute = eg_compute