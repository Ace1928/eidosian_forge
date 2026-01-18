from __future__ import absolute_import, division, print_function
import csv
import socket
import time
from string import Template
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def get_state_for(self, pxname, svname):
    """
        Find the state of specific services. When pxname is not set, get all backends for a specific host.
        Returns a list of dictionaries containing the status and weight for those services.
        """
    data = self.execute('show stat', 200, False).lstrip('# ')
    r = csv.DictReader(data.splitlines())
    state = tuple(map(lambda d: {'status': d['status'], 'weight': d['weight'], 'scur': d['scur']}, filter(lambda d: (pxname is None or d['pxname'] == pxname) and d['svname'] == svname, r)))
    return state or None