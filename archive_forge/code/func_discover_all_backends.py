from __future__ import absolute_import, division, print_function
import csv
import socket
import time
from string import Template
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def discover_all_backends(self):
    """
        Discover all entries with svname = 'BACKEND' and return a list of their corresponding
        pxnames
        """
    data = self.execute('show stat', 200, False).lstrip('# ')
    r = csv.DictReader(data.splitlines())
    return tuple(map(lambda d: d['pxname'], filter(lambda d: d['svname'] == 'BACKEND', r)))