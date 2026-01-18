from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
from ..module_utils.vendor.hcloud.servers import BoundServer
@classmethod
def define_module(cls):
    return AnsibleModule(argument_spec=dict(type={'type': 'str', 'required': True, 'choices': ['server', 'label_selector', 'ip']}, load_balancer={'type': 'str', 'required': True}, server={'type': 'str'}, label_selector={'type': 'str'}, ip={'type': 'str'}, use_private_ip={'type': 'bool', 'default': False}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), supports_check_mode=True)