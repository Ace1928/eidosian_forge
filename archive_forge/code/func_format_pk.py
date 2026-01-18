from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.api import (
from ansible_collections.community.routeros.plugins.module_utils._api_data import (
def format_pk(primary_keys, values):
    return ', '.join(('{pk}="{value}"'.format(pk=pk, value=value) for pk, value in zip(primary_keys, values)))