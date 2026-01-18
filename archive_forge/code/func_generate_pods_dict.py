from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_pods_dict(module, array):
    pods_info = {}
    api_version = array._list_available_rest_versions()
    if AC_REQUIRED_API_VERSION in api_version:
        pods = array.list_pods(mediator=True)
        for pod in range(0, len(pods)):
            acpod = pods[pod]['name']
            pods_info[acpod] = {'source': pods[pod]['source'], 'arrays': pods[pod]['arrays'], 'mediator': pods[pod]['mediator'], 'mediator_version': pods[pod]['mediator_version']}
            if ACTIVE_DR_API in api_version:
                if pods_info[acpod]['arrays'][0]['frozen_at']:
                    frozen_time = pods_info[acpod]['arrays'][0]['frozen_at'] / 1000
                    frozen_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(frozen_time))
                    pods_info[acpod]['arrays'][0]['frozen_at'] = frozen_datetime
                pods_info[acpod]['link_source_count'] = pods[pod]['link_source_count']
                pods_info[acpod]['link_target_count'] = pods[pod]['link_target_count']
                pods_info[acpod]['promotion_status'] = pods[pod]['promotion_status']
                pods_info[acpod]['requested_promotion_state'] = pods[pod]['requested_promotion_state']
        if PREFERRED_API_VERSION in api_version:
            pods_fp = array.list_pods(failover_preference=True)
            for pod in range(0, len(pods_fp)):
                acpod = pods_fp[pod]['name']
                pods_info[acpod]['failover_preference'] = pods_fp[pod]['failover_preference']
        if V6_MINIMUM_API_VERSION in api_version:
            arrayv6 = get_array(module)
            pods = list(arrayv6.get_pods(destroyed=False).items)
            for pod in range(0, len(pods)):
                name = pods[pod].name
                pods_info[name]['snapshots'] = getattr(pods[pod].space, 'snapshots', None)
                pods_info[name]['shared'] = getattr(pods[pod].space, 'shared', None)
                pods_info[name]['data_reduction'] = getattr(pods[pod].space, 'data_reduction', None)
                pods_info[name]['thin_provisioning'] = getattr(pods[pod].space, 'thin_provisioning', None)
                pods_info[name]['total_physical'] = getattr(pods[pod].space, 'total_physical', None)
                pods_info[name]['total_provisioned'] = getattr(pods[pod].space, 'total_provisioned', None)
                pods_info[name]['total_reduction'] = getattr(pods[pod].space, 'total_reduction', None)
                pods_info[name]['unique'] = getattr(pods[pod].space, 'unique', None)
                pods_info[name]['virtual'] = getattr(pods[pod].space, 'virtual', None)
                pods_info[name]['replication'] = getattr(pods[pod].space, 'replication', None)
                pods_info[name]['used_provisioned'] = getattr(pods[pod].space, 'used_provisioned', None)
                if SUBS_API_VERSION in api_version:
                    pods_info[name]['total_used'] = pods[pod].space.total_used
    return pods_info