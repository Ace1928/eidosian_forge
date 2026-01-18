from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import run_commands
from ansible_collections.community.network.plugins.module_utils.network.cnos.cnos import check_args
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import zip
def parse_image(self, data):
    match = re.search('(.*) image(.*)', data, re.M | re.I)
    if match:
        return 'Image1'
    else:
        return 'Image2'