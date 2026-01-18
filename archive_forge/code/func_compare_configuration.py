from __future__ import absolute_import, division, print_function
import json
import re
from functools import wraps
from itertools import chain
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.cliconf_base import CliconfBase
@configure
def compare_configuration(self, rollback_id=None):
    command = 'show | compare'
    if rollback_id is not None:
        command += ' rollback %s' % int(rollback_id)
    resp = self.send_command(command)
    r = resp.splitlines()
    if len(r) == 1 and '[edit]' in r[0] or (len(r) == 4 and r[1].startswith('- version')):
        resp = ''
    return resp