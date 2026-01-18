import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
@cache
def create_api_client(configuration, **headers):
    client = kubernetes.client.ApiClient(configuration)
    for header, value in headers.items():
        _set_header(client, header, value)
    return k8sdynamicclient.K8SDynamicClient(client, discoverer=LazyDiscoverer)