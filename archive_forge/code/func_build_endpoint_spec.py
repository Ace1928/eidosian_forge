from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def build_endpoint_spec(self):
    endpoint_spec_args = {}
    if self.publish is not None:
        ports = []
        for port in self.publish:
            port_spec = {'Protocol': port['protocol'], 'TargetPort': port['target_port']}
            if port.get('published_port'):
                port_spec['PublishedPort'] = port['published_port']
            if port.get('mode'):
                port_spec['PublishMode'] = port['mode']
            ports.append(port_spec)
        endpoint_spec_args['ports'] = ports
    if self.endpoint_mode is not None:
        endpoint_spec_args['mode'] = self.endpoint_mode
    return types.EndpointSpec(**endpoint_spec_args) if endpoint_spec_args else None