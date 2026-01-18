from __future__ import (absolute_import, division, print_function)
import os
import time
from ansible.module_utils.basic import AnsibleModule
def expand_integrations(eg, module):
    rancher = module.params.get('rancher')
    mesosphere = module.params.get('mesosphere')
    ecs = module.params.get('ecs')
    kubernetes = module.params.get('kubernetes')
    right_scale = module.params.get('right_scale')
    opsworks = module.params.get('opsworks')
    chef = module.params.get('chef')
    integration_exists = False
    eg_integrations = spotinst.aws_elastigroup.ThirdPartyIntegrations()
    if mesosphere is not None:
        eg_integrations.mesosphere = expand_fields(mesosphere_fields, mesosphere, 'Mesosphere')
        integration_exists = True
    if ecs is not None:
        eg_integrations.ecs = expand_fields(ecs_fields, ecs, 'EcsConfiguration')
        integration_exists = True
    if kubernetes is not None:
        eg_integrations.kubernetes = expand_fields(kubernetes_fields, kubernetes, 'KubernetesConfiguration')
        integration_exists = True
    if right_scale is not None:
        eg_integrations.right_scale = expand_fields(right_scale_fields, right_scale, 'RightScaleConfiguration')
        integration_exists = True
    if opsworks is not None:
        eg_integrations.opsworks = expand_fields(opsworks_fields, opsworks, 'OpsWorksConfiguration')
        integration_exists = True
    if rancher is not None:
        eg_integrations.rancher = expand_fields(rancher_fields, rancher, 'Rancher')
        integration_exists = True
    if chef is not None:
        eg_integrations.chef = expand_fields(chef_fields, chef, 'ChefConfiguration')
        integration_exists = True
    if integration_exists:
        eg.third_parties_integration = eg_integrations