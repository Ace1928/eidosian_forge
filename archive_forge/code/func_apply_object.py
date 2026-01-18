from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import json
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.kubernetes.core.plugins.module_utils.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
def apply_object(resource, definition, server_side=False):
    try:
        actual = resource.get(name=definition['metadata']['name'], namespace=definition['metadata'].get('namespace'))
        if server_side:
            return (actual, None)
    except NotFoundError:
        return (None, dict_merge(definition, annotate(definition)))
    return apply_patch(actual.to_dict(), definition)