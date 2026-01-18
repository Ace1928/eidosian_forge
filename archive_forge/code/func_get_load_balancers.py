from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
def get_load_balancers(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_load_balancer_info = [self.client.load_balancers.get_by_id(self.module.params.get('id'))]
        elif self.module.params.get('name') is not None:
            self.hcloud_load_balancer_info = [self.client.load_balancers.get_by_name(self.module.params.get('name'))]
        else:
            params = {}
            label_selector = self.module.params.get('label_selector')
            if label_selector:
                params['label_selector'] = label_selector
            self.hcloud_load_balancer_info = self.client.load_balancers.get_all(**params)
    except HCloudException as exception:
        self.fail_json_hcloud(exception)