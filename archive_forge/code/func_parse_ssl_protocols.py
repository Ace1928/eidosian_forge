from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import Version
def parse_ssl_protocols(data):
    tlsv1_0 = re.search('(?<!\\S)TLSv1(?!\\S)', data, re.M) is not None
    tlsv1_1 = re.search('(?<!\\S)TLSv1.1(?!\\S)', data, re.M) is not None
    tlsv1_2 = re.search('(?<!\\S)TLSv1.2(?!\\S)', data, re.M) is not None
    return {'tlsv1_0': tlsv1_0, 'tlsv1_1': tlsv1_1, 'tlsv1_2': tlsv1_2}