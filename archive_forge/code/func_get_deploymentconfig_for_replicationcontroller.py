from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone
import traceback
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
def get_deploymentconfig_for_replicationcontroller(replica_controller):
    DeploymentConfigAnnotation = 'openshift.io/deployment-config.name'
    try:
        deploymentconfig_name = replica_controller['metadata']['annotations'].get(DeploymentConfigAnnotation)
        if deploymentconfig_name is None or deploymentconfig_name == '':
            return None
        return deploymentconfig_name
    except Exception:
        return None