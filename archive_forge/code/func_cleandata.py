from __future__ import (absolute_import, division, print_function)
import json
import re
import time
import os
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.six import raise_from
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def cleandata(self):
    """Clean the dynamic inventory

        The first version of the inventory only supported container.
        This will change in the future.
        The following function cleans up the data and remove the all items with the wrong type.

        Args:
            None
        Kwargs:
            None
        Raises:
            None
        Returns:
            None"""
    iter_keys = list(self.data['instances'].keys())
    for instance_name in iter_keys:
        if self._get_data_entry('instances/{0}/instances/metadata/type'.format(instance_name)) != self.type_filter:
            del self.data['instances'][instance_name]