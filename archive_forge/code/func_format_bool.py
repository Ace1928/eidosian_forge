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
def format_bool(v):
    return 'yes' if v else 'no'