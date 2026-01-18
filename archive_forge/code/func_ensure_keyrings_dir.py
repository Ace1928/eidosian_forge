from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import textwrap
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import raise_from  # type: ignore[attr-defined]
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import open_url
from ansible.module_utils.urls import get_user_agent
from ansible.module_utils.urls import urlparse
def ensure_keyrings_dir(module):
    changed = False
    if not os.path.isdir(KEYRINGS_DIR):
        if not module.check_mode:
            os.mkdir(KEYRINGS_DIR, 493)
        changed |= True
    changed |= module.set_fs_attributes_if_different({'path': KEYRINGS_DIR, 'secontext': [None, None, None], 'owner': 'root', 'group': 'root', 'mode': '0755', 'attributes': None}, changed)
    return changed