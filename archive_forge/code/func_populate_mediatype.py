from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import run_commands
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
def populate_mediatype(self, data):
    lines = data.split('\n')
    for line in lines:
        match = re.match('Port (\\S+):\\W+Type\\W+:\\W+(.*)', line)
        if match:
            self.facts['interfaces'][match.group(1)]['mediatype'] = match.group(2)