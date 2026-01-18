from __future__ import absolute_import, division, print_function
import io
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def match_opt(option, line):
    option = re.escape(option)
    return re.match('([#;]?)( |\t)*(%s)( |\t)*(=|$)( |\t)*(.*)' % option, line)