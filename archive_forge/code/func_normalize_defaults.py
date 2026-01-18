from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def normalize_defaults(self):
    """Update ref defaults with normalized data"""
    ref = self._ref
    for k in ref['commands']:
        if 'default' in ref[k] and ref[k]['default']:
            kind = ref[k]['kind']
            if 'int' == kind:
                ref[k]['default'] = int(ref[k]['default'])
            elif 'list' == kind:
                ref[k]['default'] = [str(i) for i in ref[k]['default']]
            elif 'dict' == kind:
                for key, v in ref[k]['default'].items():
                    if v:
                        v = str(v)
                    ref[k]['default'][key] = v