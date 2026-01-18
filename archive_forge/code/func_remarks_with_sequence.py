from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def remarks_with_sequence(remarks_data):
    cmd = 'remark '
    if remarks_data.get('remarks'):
        cmd += remarks_data.get('remarks')
    if remarks_data.get('sequence'):
        cmd = to_text(remarks_data.get('sequence')) + ' ' + cmd
    return cmd