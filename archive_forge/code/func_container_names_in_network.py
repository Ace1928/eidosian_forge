from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def container_names_in_network(network):
    return [c['Name'] for c in network['Containers'].values()] if network['Containers'] else []