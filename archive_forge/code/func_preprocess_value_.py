from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def preprocess_value_(module, client, api_version, options, values):
    if len(options) != 1:
        raise AssertionError('host_config_value can only be used for a single option')
    if preprocess_value is not None and options[0].name in values:
        value = preprocess_value(module, client, api_version, values[options[0].name])
        if value is None:
            del values[options[0].name]
        else:
            values[options[0].name] = value
    return values