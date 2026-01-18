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
@staticmethod
def load_json_data(path):
    """Load json data

        Load json data from file

        Args:
            list(path): Path elements
            str(file_name): Filename of data
        Kwargs:
            None
        Raises:
            None
        Returns:
            dict(json_data): json data"""
    try:
        with open(path, 'r') as json_file:
            return json.load(json_file)
    except (IOError, json.decoder.JSONDecodeError) as err:
        raise AnsibleParserError('Could not load the test data from {0}: {1}'.format(to_native(path), to_native(err)))