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
def build_docker_service(self):
    container_spec = self.build_container_spec()
    placement = self.build_placement()
    task_template = self.build_task_template(container_spec, placement)
    update_config = self.build_update_config()
    rollback_config = self.build_rollback_config()
    service_mode = self.build_service_mode()
    endpoint_spec = self.build_endpoint_spec()
    service = {'task_template': task_template, 'mode': service_mode}
    if update_config:
        service['update_config'] = update_config
    if rollback_config:
        service['rollback_config'] = rollback_config
    if endpoint_spec:
        service['endpoint_spec'] = endpoint_spec
    if self.labels:
        service['labels'] = self.labels
    if not self.can_use_task_template_networks:
        networks = self.build_networks()
        if networks:
            service['networks'] = networks
    return service